"""BF16 packed QKV RoPE for G4: preserve eager [even | odd] layout and BF16 rounding."""
import torch
import triton
import triton.language as tl
from pymss_core.modules.bs_roformer import transformer as tr


@triton.jit
def _fwd(P, C, S, Q, K, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, L: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    which = tl.program_id(1)
    j = tl.arange(0, L)
    u = j // (D//2) * D + (j % (D//2)) * 2
    off = (row * 3 + which) * H * D + u
    x = tl.load(P+off); y = tl.load(P+off+1)
    t = row % N * (D//2) + j % (D//2)
    c = tl.load(C+t); s = tl.load(S+t)
    e = (x*c).to(tl.bfloat16) - (y*s).to(tl.bfloat16)
    o = (y*c).to(tl.bfloat16) + (x*s).to(tl.bfloat16)
    dst = row*H*D + j//(D//2)*D + j%(D//2)
    if which == 0:
        tl.store(Q+dst,e); tl.store(Q+dst+D//2,o)
    else:
        tl.store(K+dst,e); tl.store(K+dst+D//2,o)


@triton.jit
def _bwd(Q,K,V,P,C,S,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,L:tl.constexpr,
         QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr):
    row=tl.program_id(0).to(tl.int64); which=tl.program_id(1)
    b=row//N; n=row%N; j=tl.arange(0,L); h=j//(D//2); d=j%(D//2)
    dst=P+(row*3+which)*H*D+h*D+2*d
    if which==2:
        v=V+b*VS[0]+n*VS[1]+h*VS[2]+2*d*VS[3]
        tl.store(dst,tl.load(v)); tl.store(dst+1,tl.load(v+VS[3]))
    else:
        if which==0:
            p=Q+b*QS[0]+n*QS[1]+h*QS[2]+d*QS[3]
            e=tl.load(p); o=tl.load(p+(D//2)*QS[3])
        else:
            p=K+b*KS[0]+n*KS[1]+h*KS[2]+d*KS[3]
            e=tl.load(p); o=tl.load(p+(D//2)*KS[3])
        c=tl.load(C+n*(D//2)+d); s=tl.load(S+n*(D//2)+d)
        tl.store(dst,(e*c).to(tl.bfloat16)+(o*s).to(tl.bfloat16))
        tl.store(dst+1,(o*c).to(tl.bfloat16)-(e*s).to(tl.bfloat16))


class Packed(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p,c,s,h):
        b,n,w=p.shape; d=w//(3*h); l=h*d//2
        assert w==3*h*d and d%2==0 and l==triton.next_power_of_2(l)
        assert p.is_cuda and p.dtype==c.dtype==s.dtype==torch.bfloat16 and p.is_contiguous()
        assert c.shape==s.shape==(1,n,1,d//2) and c.is_contiguous() and s.is_contiguous()
        assert not c.requires_grad and not s.requires_grad
        q=p.new_empty((b,n,h,d)); k=torch.empty_like(q)
        with torch.cuda.device(p.device): _fwd[(b*n,2)](p,c,s,q,k,n,h,d,l,enable_fp_fusion=False)
        ctx.save_for_backward(c,s);ctx.sizes=(b,n,h,d)
        return q,k,p.view(b,n,3,h,d)[:,:,2]

    @staticmethod
    def backward(ctx,q,k,v):
        c,s=ctx.saved_tensors;b,n,h,d=ctx.sizes;p=q.new_empty((b,n,3*h*d))
        with torch.cuda.device(p.device): _bwd[(b*n,3)](q,k,v,p,c,s,n,h,d,h*d//2,q.stride(),k.stride(),v.stride(),enable_fp_fusion=False)
        return p,None,None,None


@torch.compile(backend="inductor", fullgraph=True)
def _attention_gate(attn, logits):
    return (attn.transpose(1, 2) * logits.sigmoid().unsqueeze(-1)).flatten(-2).contiguous()


class PackedAttention(tr.Attention):
    def forward(self,x):
        if not (self.training and torch.is_grad_enabled() and x.is_cuda and torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')==torch.bfloat16):
            return super().forward(x)
        x=self.norm(x); p=self.to_qkv(x); n=p.shape[1]
        cs,sn=tr.cached_rotary_cos_sin(self.rotary_embed,n,p.device,p.dtype)
        q,k,v=Packed.apply(p,cs,sn,self.heads)
        out=self._attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
        return self.to_out(_attention_gate(out, self.to_gates(x)))


def install(model):
    count=0
    for mod in model.modules():
        if type(mod) is tr.Attention and mod.rotary_embed is not None and not any(p.requires_grad for p in mod.rotary_embed.parameters()) and not mod.use_pope and mod.heads==8 and mod.to_qkv.out_features==1536:
            mod.__class__=PackedAttention; count+=1
    return count
