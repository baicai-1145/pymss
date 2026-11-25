"""Experimental G4 temporal depthwise convolution. Layout is [B,T,F,C]."""
import torch
import triton
import triton.language as tl
from pymss_core.modules.bs_roformer.transformer import ConvFFN


@triton.jit
def _time_filter(X, W, Bias, Y, T:tl.constexpr, F:tl.constexpr, C:tl.constexpr,
                 K:tl.constexpr, BACK:tl.constexpr, BT:tl.constexpr, BC:tl.constexpr):
    t = tl.program_id(0)*BT + tl.arange(0,BT)
    c = tl.program_id(1)*BC + tl.arange(0,BC)
    bf = tl.program_id(2); b=bf//F; f=bf%F
    ptr=((b*T+t[:,None])*F+f)*C+c[None,:]
    acc=tl.full((BT,BC),0,tl.float32)
    for j in tl.static_range(K):
        dt=j-K//2
        idx=K-1-j if BACK else j
        a=tl.load(X+ptr+dt*F*C, ((t+dt>=0)&(t+dt<T))[:,None] & (c<C)[None,:],0)
        w=tl.load(W+c*K+idx,c<C,0)
        acc=tl.fma(a.to(tl.float32),w[None,:].to(tl.float32),acc)
    if not BACK: acc+=tl.load(Bias+c,c<C,0)[None,:].to(tl.float32)
    tl.store(Y+ptr,acc,(t<T)[:,None] & (c<C)[None,:])


@triton.jit
def _weight_partials(X,G,P,T:tl.constexpr,F:tl.constexpr,C:tl.constexpr,K:tl.constexpr,
                     ROWS:tl.constexpr,BR:tl.constexpr,BC:tl.constexpr):
    part=tl.program_id(0); r=part*BR+tl.arange(0,BR); c=tl.program_id(1)*BC+tl.arange(0,BC)
    t=r//F%T; p=r[:,None]*C+c[None,:]; valid=(r<ROWS)[:,None] & (c<C)[None,:]
    dy=tl.load(G+p,valid,0).to(tl.float32)
    tl.store(P+(part*(K+1)+K)*C+c,tl.sum(dy,0),c<C)
    for j in range(K):
        dt=j-K//2
        x=tl.load(X+p+dt*F*C,valid & ((t+dt>=0)&(t+dt<T))[:,None],0).to(tl.float32)
        tl.store(P+(part*(K+1)+j)*C+c,tl.sum(x*dy,0),c<C)


class TimeDW(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,bias):
        b,t,f,c=x.shape;k=w.shape[-1]; assert w.shape==(c,1,k) and k%2==1
        assert x.is_cuda and x.dtype==w.dtype==bias.dtype==torch.bfloat16
        x=x.contiguous();w=w.contiguous(); y=torch.empty_like(x)
        with torch.cuda.device(x.device):
            _time_filter[(triton.cdiv(t,16),triton.cdiv(c,64),b*f)](x,w,bias,y,t,f,c,k,False,16,64)
        ctx.save_for_backward(x,w);return y

    @staticmethod
    def backward(ctx,g):
        x,w=ctx.saved_tensors;b,t,f,c=x.shape;k=w.shape[-1];g=g.contiguous();dx=torch.empty_like(x)
        n=triton.cdiv(b*t*f,128);parts=x.new_empty((n,k+1,c),dtype=torch.float32)
        with torch.cuda.device(x.device):
            _time_filter[(triton.cdiv(t,16),triton.cdiv(c,64),b*f)](g,w,w,dx,t,f,c,k,True,16,64)
            _weight_partials[(n,triton.cdiv(c,32))](x,g,parts,t,f,c,k,b*t*f,128,32)
        s=parts.sum(0).to(w.dtype)
        return dx,s[:-1].T.contiguous().view_as(w),s[-1]


@torch.compile(backend="inductor", fullgraph=True)
def _time_gate(projected):
    u, v = projected.chunk(2, dim=-1)
    return u * v.sigmoid()


@triton.jit
def _dx_gate(G,W,P,DP,T:tl.constexpr,F:tl.constexpr,C:tl.constexpr,K:tl.constexpr,BT:tl.constexpr,BC:tl.constexpr):
    t=tl.program_id(0)*BT+tl.arange(0,BT);c=tl.program_id(1)*BC+tl.arange(0,BC)
    bf=tl.program_id(2); row=((bf//F)*T+t[:,None])*F+bf%F; valid=(t<T)[:,None] & (c<C)[None,:]
    acc=tl.full((BT,BC),0,tl.float32)
    for j in tl.static_range(K):
        dt=j-K//2;mask=((t+dt>=0)&(t+dt<T))[:,None] & (c<C)[None,:]
        dy=tl.load(G+(row+dt*F)*C+c[None,:],mask,0)
        w=tl.load(W+c*K+K-1-j,c<C,0)
        acc=tl.fma(dy.to(tl.float32),w[None,:].to(tl.float32),acc)
    # Match the BF16 boundary between convolution backward and sigmoid gating.
    off=row*(2*C)+c[None,:];u=tl.load(P+off,valid,0).to(tl.float32);v=tl.load(P+off+C,valid,0).to(tl.float32)
    sig=tl.sigmoid(v).to(tl.bfloat16).to(tl.float32);dh=acc.to(tl.bfloat16).to(tl.float32)
    tl.store(DP+off,dh*sig,valid);tl.store(DP+off+C,(dh*u).to(tl.bfloat16).to(tl.float32)*sig*(1.-sig),valid)


class GatedTimeDW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, weight, bias):
        h = _time_gate(projected).contiguous()
        b,t,f,c = h.shape; k = weight.shape[-1]; y = torch.empty_like(h)
        with torch.cuda.device(h.device):
            _time_filter[(triton.cdiv(t,16),triton.cdiv(c,64),b*f)](h,weight,bias,y,t,f,c,k,False,16,64)
        ctx.save_for_backward(projected,h,weight)
        return y

    @staticmethod
    def backward(ctx, grad):
        p,h,w = ctx.saved_tensors; b,t,f,c = h.shape; k = w.shape[-1]
        dy = grad.contiguous(); dp = torch.empty_like(p)
        n = triton.cdiv(b*t*f,128); part = h.new_empty((n,k+1,c),dtype=torch.float32)
        with torch.cuda.device(h.device):
            _dx_gate[(triton.cdiv(t,16),triton.cdiv(c,64),b*f)](dy,w,p,dp,t,f,c,k,16,64)
            _weight_partials[(n,triton.cdiv(c,32))](h,dy,part,t,f,c,k,b*t*f,128,32)
        reduced = part.sum(0).to(w.dtype)
        return dp,reduced[:-1].T.contiguous().view_as(w),reduced[-1]


class TimeConv(ConvFFN):
    def forward(self,x):
        b,t,f,_ = x.shape
        c = self.conv.in_channels
        if not (self.training and torch.is_grad_enabled() and x.is_cuda
                and torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.bfloat16
                and b*f <= 65535 and (b*t + 32)*f*2*c < 2**31):
            return super().forward(x)
        projected = self.proj_in(x).contiguous()
        c = self.conv
        h = GatedTimeDW.apply(projected,c.weight.to(projected.dtype).contiguous(),c.bias.to(projected.dtype))
        return self.drop(self.proj_out(h))


def install(m):
    count=0
    for mod in m.modules():
        if type(mod) is ConvFFN:
            c = mod.conv
            if not (c.stride == c.dilation == (1,) and c.padding == (15,) and c.kernel_size == (31,)
                    and c.groups == c.in_channels == c.out_channels and c.bias is not None
                    and c.padding_mode == 'zeros'):
                continue
            mod.__class__=TimeConv;count+=1
    return count
