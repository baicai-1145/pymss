import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


__all__ = (
    'attention_gate_out_inplace_triton',
    'attention_gate_out_rope_inplace_triton',
    'attention_gate_packed64_5090_triton',
    'attention_gate_triton',
    'attention_gate_varlen_triton',
    'attention_gate_out_triton',
    'copy_rms_norm_4d_triton',
    'mask_final_tanh_glu_flat_triton',
    'mask_final_tanh_glu_packed_triton',
    'rms_norm_triton',
    'rotate_qk_inplace_triton',
    'triton_kernels_available',
)


def triton_kernels_available():
    return triton is not None and tl is not None


if triton is not None:
    _ATTENTION_GATE_CONFIGS = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ]
    _ATTENTION_GATE_PACKED64_5090_CONFIGS = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ]
    _ATTENTION_GATE_OUT_CONFIGS = [
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_O': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_O': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_O': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_O': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_O': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_O': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_O': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_O': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_O': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_O': 128}, num_warps=4, num_stages=3),
    ]
    @triton.jit
    def _rms_norm_kernel(
            x_ptr,
            gamma_ptr,
            out_ptr,
            rows,
            dim: tl.constexpr,
            eps: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < dim
        x = tl.load(x_ptr + row * dim + offs, mask=mask, other=0.).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.).to(tl.float32)
        x = x * tl.rsqrt(tl.sum(x * x, axis=0) / dim + eps) * gamma
        tl.store(out_ptr + row * dim + offs, x, mask=mask)

    @triton.jit
    def _rms_norm_rows_kernel(
            x_ptr,
            gamma_ptr,
            out_ptr,
            rows,
            dim: tl.constexpr,
            eps: tl.constexpr,
            BLOCK_R: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        offs = tl.arange(0, BLOCK_D)
        mask = (row[:, None] < rows) & (offs[None, :] < dim)
        x = tl.load(x_ptr + row[:, None] * dim + offs[None, :], mask=mask, other=0.).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs, mask=offs < dim, other=0.).to(tl.float32)
        x = x * tl.rsqrt(tl.sum(x * x, axis=1)[:, None] / dim + eps) * gamma[None, :]
        tl.store(out_ptr + row[:, None] * dim + offs[None, :], x, mask=mask)

    @triton.jit
    def _copy_rms_norm_4d_kernel(
            x_ptr,
            gamma_ptr,
            normed_ptr,
            residual_ptr,
            rows,
            size_b: tl.constexpr,
            size_t: tl.constexpr,
            size_f: tl.constexpr,
            dim: tl.constexpr,
            x_b_stride,
            x_t_stride,
            x_f_stride,
            x_d_stride,
            eps: tl.constexpr,
            ORDER: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        d_mask = offs < dim

        if ORDER == 0:
            batch = row // (size_f * size_t)
            rem = row - batch * size_f * size_t
            freq = rem // size_t
            time = rem - freq * size_t
        else:
            batch = row // (size_t * size_f)
            rem = row - batch * size_t * size_f
            time = rem // size_f
            freq = rem - time * size_f

        x = tl.load(
            x_ptr + batch * x_b_stride + time * x_t_stride + freq * x_f_stride + offs * x_d_stride,
            mask=d_mask,
            other=0.,
        )
        gamma = tl.load(gamma_ptr + offs, mask=d_mask, other=0.).to(tl.float32)
        scale = tl.rsqrt(tl.sum(x.to(tl.float32) * x.to(tl.float32), axis=0) / dim + eps)
        tl.store(residual_ptr + row * dim + offs, x, mask=d_mask)
        tl.store(normed_ptr + row * dim + offs, x.to(tl.float32) * scale * gamma, mask=d_mask)

    @triton.jit
    def _rotate_qk_inplace_kernel(
            q_ptr,
            k_ptr,
            cos_ptr,
            sin_ptr,
            total_pairs,
            q_b_stride,
            q_t_stride,
            q_h_stride,
            q_d_stride,
            k_b_stride,
            k_t_stride,
            k_h_stride,
            k_d_stride,
            cos_t_stride,
            cos_d_stride,
            sin_t_stride,
            sin_d_stride,
            seq_len: tl.constexpr,
            heads: tl.constexpr,
            half_dim: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_pairs
        pair = offsets % half_dim
        head = (offsets // half_dim) % heads
        time = (offsets // (half_dim * heads)) % seq_len
        batch = offsets // (half_dim * heads * seq_len)
        dim = pair * 2
        q_base = batch * q_b_stride + time * q_t_stride + head * q_h_stride + dim * q_d_stride
        k_base = batch * k_b_stride + time * k_t_stride + head * k_h_stride + dim * k_d_stride
        cos = tl.load(cos_ptr + time * cos_t_stride + dim * cos_d_stride, mask=mask, other=1.)
        sin = tl.load(sin_ptr + time * sin_t_stride + dim * sin_d_stride, mask=mask, other=0.)

        q_real = tl.load(q_ptr + q_base, mask=mask, other=0.)
        q_imag = tl.load(q_ptr + q_base + q_d_stride, mask=mask, other=0.)
        k_real = tl.load(k_ptr + k_base, mask=mask, other=0.)
        k_imag = tl.load(k_ptr + k_base + k_d_stride, mask=mask, other=0.)

        tl.store(q_ptr + q_base, q_real * cos - q_imag * sin, mask=mask)
        tl.store(q_ptr + q_base + q_d_stride, q_imag * cos + q_real * sin, mask=mask)
        tl.store(k_ptr + k_base, k_real * cos - k_imag * sin, mask=mask)
        tl.store(k_ptr + k_base + k_d_stride, k_imag * cos + k_real * sin, mask=mask)

    @triton.jit
    def _init_linear_out_kernel(
            bias_ptr,
            residual_ptr,
            dst_ptr,
            total,
            seq_len: tl.constexpr,
            out_dim: tl.constexpr,
            residual_b_stride,
            residual_t_stride,
            residual_n_stride,
            HAS_BIAS: tl.constexpr,
            HAS_RESIDUAL: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        n = offsets % out_dim
        t = (offsets // out_dim) % seq_len
        b = offsets // (seq_len * out_dim)
        value = tl.load(bias_ptr + n, mask=mask, other=0.).to(tl.float32) if HAS_BIAS else tl.zeros((BLOCK_SIZE,), tl.float32)
        if HAS_RESIDUAL:
            value += tl.load(
                residual_ptr + b * residual_b_stride + t * residual_t_stride + n * residual_n_stride,
                mask=mask,
                other=0.,
            ).to(tl.float32)
        tl.store(dst_ptr + offsets, value, mask=mask)

    @triton.autotune(configs=_ATTENTION_GATE_CONFIGS, key=['seq_len', 'dim_head'])
    @triton.jit
    def _attention_gate_kernel(
            q_ptr,
            k_ptr,
            v_ptr,
            gates_ptr,
            dst_ptr,
            q_b_stride,
            q_h_stride,
            q_t_stride,
            q_d_stride,
            k_b_stride,
            k_h_stride,
            k_t_stride,
            k_d_stride,
            v_b_stride,
            v_h_stride,
            v_t_stride,
            v_d_stride,
            gates_b_stride,
            gates_t_stride,
            gates_h_stride,
            dst_b_stride,
            dst_t_stride,
            dst_d_stride,
            scale_log2e,
            seq_len: tl.constexpr,
            heads: tl.constexpr,
            dim_head: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch = pid_bh // heads
        head = pid_bh - batch * heads
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(
            q_ptr + batch * q_b_stride + head * q_h_stride + offs_m[:, None] * q_t_stride + offs_d[None, :] * q_d_stride,
            mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < dim_head),
            other=0.,
        )
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            cols = start_n + offs_n
            k = tl.load(
                k_ptr + batch * k_b_stride + head * k_h_stride + cols[:, None] * k_t_stride + offs_d[None, :] * k_d_stride,
                mask=(cols[:, None] < seq_len) & (offs_d[None, :] < dim_head),
                other=0.,
            )
            v = tl.load(
                v_ptr + batch * v_b_stride + head * v_h_stride + cols[:, None] * v_t_stride + offs_d[None, :] * v_d_stride,
                mask=(cols[:, None] < seq_len) & (offs_d[None, :] < dim_head),
                other=0.,
            )
            qk = tl.dot(q, tl.trans(k)) * scale_log2e
            qk = tl.where((offs_m[:, None] < seq_len) & (cols[None, :] < seq_len), qk, -float('inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        gate = tl.load(
            gates_ptr + batch * gates_b_stride + offs_m * gates_t_stride + head * gates_h_stride,
            mask=offs_m < seq_len,
            other=-float('inf'),
        ).to(tl.float32)
        out = (acc / l_i[:, None]) * tl.sigmoid(gate)[:, None]
        tl.store(
            dst_ptr + batch * dst_b_stride + offs_m[:, None] * dst_t_stride + (head * dim_head + offs_d)[None, :] * dst_d_stride,
            out,
            mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < dim_head),
        )

    @triton.autotune(configs=_ATTENTION_GATE_CONFIGS, key=['q_len', 'kv_len', 'dim_head'])
    @triton.jit
    def _attention_gate_varlen_kernel(
            q_ptr,
            k_ptr,
            v_ptr,
            gates_ptr,
            dst_ptr,
            q_b_stride,
            q_h_stride,
            q_t_stride,
            q_d_stride,
            k_b_stride,
            k_h_stride,
            k_t_stride,
            k_d_stride,
            v_b_stride,
            v_h_stride,
            v_t_stride,
            v_d_stride,
            gates_b_stride,
            gates_t_stride,
            gates_h_stride,
            dst_b_stride,
            dst_t_stride,
            dst_d_stride,
            scale_log2e,
            q_len: tl.constexpr,
            kv_len: tl.constexpr,
            heads: tl.constexpr,
            dim_head: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch = pid_bh // heads
        head = pid_bh - batch * heads
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(
            q_ptr + batch * q_b_stride + head * q_h_stride + offs_m[:, None] * q_t_stride + offs_d[None, :] * q_d_stride,
            mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < dim_head),
            other=0.,
        )
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)

        for start_n in range(0, kv_len, BLOCK_N):
            cols = start_n + offs_n
            k = tl.load(
                k_ptr + batch * k_b_stride + head * k_h_stride + cols[:, None] * k_t_stride + offs_d[None, :] * k_d_stride,
                mask=(cols[:, None] < kv_len) & (offs_d[None, :] < dim_head),
                other=0.,
            )
            v = tl.load(
                v_ptr + batch * v_b_stride + head * v_h_stride + cols[:, None] * v_t_stride + offs_d[None, :] * v_d_stride,
                mask=(cols[:, None] < kv_len) & (offs_d[None, :] < dim_head),
                other=0.,
            )
            qk = tl.dot(q, tl.trans(k)) * scale_log2e
            qk = tl.where((offs_m[:, None] < q_len) & (cols[None, :] < kv_len), qk, -float('inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        gate = tl.load(
            gates_ptr + batch * gates_b_stride + offs_m * gates_t_stride + head * gates_h_stride,
            mask=offs_m < q_len,
            other=-float('inf'),
        ).to(tl.float32)
        out = (acc / l_i[:, None]) * tl.sigmoid(gate)[:, None]
        tl.store(
            dst_ptr + batch * dst_b_stride + offs_m[:, None] * dst_t_stride + (head * dim_head + offs_d)[None, :] * dst_d_stride,
            out,
            mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < dim_head),
        )

    @triton.autotune(configs=_ATTENTION_GATE_PACKED64_5090_CONFIGS, key=['seq_len'])
    @triton.jit
    def _attention_gate_packed64_5090_kernel(
            q_ptr,
            k_ptr,
            v_ptr,
            gates_ptr,
            dst_ptr,
            scale_log2e,
            seq_len: tl.constexpr,
            heads: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch = pid_bh // heads
        head = pid_bh - batch * heads
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, 64)
        packed_t_stride = 3 * heads * 64
        packed_b_stride = seq_len * packed_t_stride
        out_t_stride = heads * 64

        q = tl.load(
            q_ptr + batch * packed_b_stride + head * 64 + offs_m[:, None] * packed_t_stride + offs_d[None, :],
            mask=offs_m[:, None] < seq_len,
            other=0.,
        )
        acc = tl.zeros((BLOCK_M, 64), tl.float32)
        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            cols = start_n + offs_n
            k = tl.load(
                k_ptr + batch * packed_b_stride + head * 64 + cols[:, None] * packed_t_stride + offs_d[None, :],
                mask=cols[:, None] < seq_len,
                other=0.,
            )
            v = tl.load(
                v_ptr + batch * packed_b_stride + head * 64 + cols[:, None] * packed_t_stride + offs_d[None, :],
                mask=cols[:, None] < seq_len,
                other=0.,
            )
            qk = tl.dot(q, tl.trans(k)) * scale_log2e
            qk = tl.where((offs_m[:, None] < seq_len) & (cols[None, :] < seq_len), qk, -float('inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        gate = tl.load(
            gates_ptr + batch * seq_len * heads + offs_m * heads + head,
            mask=offs_m < seq_len,
            other=-float('inf'),
        ).to(tl.float32)
        out = (acc / l_i[:, None]) * tl.sigmoid(gate)[:, None]
        tl.store(
            dst_ptr + batch * seq_len * out_t_stride + offs_m[:, None] * out_t_stride + head * 64 + offs_d[None, :],
            out,
            mask=offs_m[:, None] < seq_len,
        )

    @triton.autotune(configs=_ATTENTION_GATE_OUT_CONFIGS, key=['seq_len', 'dim_head', 'out_dim'], restore_value=['dst_ptr'])
    @triton.jit
    def _attention_gate_out_kernel(
            q_ptr,
            k_ptr,
            v_ptr,
            gates_ptr,
            weight_ptr,
            dst_ptr,
            q_b_stride,
            q_h_stride,
            q_t_stride,
            q_d_stride,
            k_b_stride,
            k_h_stride,
            k_t_stride,
            k_d_stride,
            v_b_stride,
            v_h_stride,
            v_t_stride,
            v_d_stride,
            gates_b_stride,
            gates_t_stride,
            gates_h_stride,
            weight_n_stride,
            weight_k_stride,
            dst_b_stride,
            dst_t_stride,
            dst_n_stride,
            scale_log2e,
            seq_len: tl.constexpr,
            heads: tl.constexpr,
            dim_head: tl.constexpr,
            out_dim: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_O: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch = pid_bh // heads
        head = pid_bh - batch * heads
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(
            q_ptr + batch * q_b_stride + head * q_h_stride + offs_m[:, None] * q_t_stride + offs_d[None, :] * q_d_stride,
            mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < dim_head),
            other=0.,
        )
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            cols = start_n + offs_n
            k = tl.load(
                k_ptr + batch * k_b_stride + head * k_h_stride + cols[:, None] * k_t_stride + offs_d[None, :] * k_d_stride,
                mask=(cols[:, None] < seq_len) & (offs_d[None, :] < dim_head),
                other=0.,
            )
            v = tl.load(
                v_ptr + batch * v_b_stride + head * v_h_stride + cols[:, None] * v_t_stride + offs_d[None, :] * v_d_stride,
                mask=(cols[:, None] < seq_len) & (offs_d[None, :] < dim_head),
                other=0.,
            )
            qk = tl.dot(q, tl.trans(k)) * scale_log2e
            qk = tl.where((offs_m[:, None] < seq_len) & (cols[None, :] < seq_len), qk, -float('inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        gate = tl.load(
            gates_ptr + batch * gates_b_stride + offs_m * gates_t_stride + head * gates_h_stride,
            mask=offs_m < seq_len,
            other=-float('inf'),
        ).to(tl.float32)
        lhs = ((acc / l_i[:, None]) * tl.sigmoid(gate)[:, None]).to(tl.float16)

        for out_start in range(0, out_dim, BLOCK_O):
            offs_o = out_start + tl.arange(0, BLOCK_O)
            weight = tl.load(
                weight_ptr + offs_o[None, :] * weight_n_stride + (head * dim_head + offs_d)[:, None] * weight_k_stride,
                mask=(offs_o[None, :] < out_dim) & (offs_d[:, None] < dim_head),
                other=0.,
            )
            contrib = tl.dot(lhs, weight)
            tl.atomic_add(
                dst_ptr + batch * dst_b_stride + offs_m[:, None] * dst_t_stride + offs_o[None, :] * dst_n_stride,
                contrib,
                sem='relaxed',
                mask=(offs_m[:, None] < seq_len) & (offs_o[None, :] < out_dim),
            )

    @triton.autotune(configs=_ATTENTION_GATE_OUT_CONFIGS, key=['seq_len', 'dim_head', 'out_dim'], restore_value=['dst_ptr'])
    @triton.jit
    def _attention_gate_out_rope_kernel(
            q_ptr,
            k_ptr,
            v_ptr,
            gates_ptr,
            weight_ptr,
            cos_ptr,
            sin_ptr,
            bias_ptr,
            dst_ptr,
            q_b_stride,
            q_h_stride,
            q_t_stride,
            q_d_stride,
            k_b_stride,
            k_h_stride,
            k_t_stride,
            k_d_stride,
            v_b_stride,
            v_h_stride,
            v_t_stride,
            v_d_stride,
            gates_b_stride,
            gates_t_stride,
            gates_h_stride,
            weight_n_stride,
            weight_k_stride,
            cos_t_stride,
            cos_d_stride,
            sin_t_stride,
            sin_d_stride,
            dst_b_stride,
            dst_t_stride,
            dst_n_stride,
            scale_log2e,
            seq_len: tl.constexpr,
            heads: tl.constexpr,
            dim_head: tl.constexpr,
            out_dim: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_O: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch = pid_bh // heads
        head = pid_bh - batch * heads
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        pair_d = tl.where((offs_d % 2) == 0, offs_d + 1, offs_d - 1)
        even_d = (offs_d // 2) * 2
        d_mask = offs_d < dim_head
        pair_mask = pair_d < dim_head
        is_even = (offs_d % 2) == 0

        q_raw = tl.load(
            q_ptr + batch * q_b_stride + head * q_h_stride + offs_m[:, None] * q_t_stride + offs_d[None, :] * q_d_stride,
            mask=(offs_m[:, None] < seq_len) & d_mask[None, :],
            other=0.,
        )
        q_pair = tl.load(
            q_ptr + batch * q_b_stride + head * q_h_stride + offs_m[:, None] * q_t_stride + pair_d[None, :] * q_d_stride,
            mask=(offs_m[:, None] < seq_len) & pair_mask[None, :],
            other=0.,
        )
        q_cos = tl.load(
            cos_ptr + offs_m[:, None] * cos_t_stride + even_d[None, :] * cos_d_stride,
            mask=(offs_m[:, None] < seq_len) & d_mask[None, :],
            other=1.,
        )
        q_sin = tl.load(
            sin_ptr + offs_m[:, None] * sin_t_stride + even_d[None, :] * sin_d_stride,
            mask=(offs_m[:, None] < seq_len) & d_mask[None, :],
            other=0.,
        )
        q = tl.where(is_even[None, :], q_raw * q_cos - q_pair * q_sin, q_raw * q_cos + q_pair * q_sin).to(tl.float16)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            cols = start_n + offs_n
            k_raw = tl.load(
                k_ptr + batch * k_b_stride + head * k_h_stride + cols[:, None] * k_t_stride + offs_d[None, :] * k_d_stride,
                mask=(cols[:, None] < seq_len) & d_mask[None, :],
                other=0.,
            )
            k_pair = tl.load(
                k_ptr + batch * k_b_stride + head * k_h_stride + cols[:, None] * k_t_stride + pair_d[None, :] * k_d_stride,
                mask=(cols[:, None] < seq_len) & pair_mask[None, :],
                other=0.,
            )
            k_cos = tl.load(
                cos_ptr + cols[:, None] * cos_t_stride + even_d[None, :] * cos_d_stride,
                mask=(cols[:, None] < seq_len) & d_mask[None, :],
                other=1.,
            )
            k_sin = tl.load(
                sin_ptr + cols[:, None] * sin_t_stride + even_d[None, :] * sin_d_stride,
                mask=(cols[:, None] < seq_len) & d_mask[None, :],
                other=0.,
            )
            k = tl.where(is_even[None, :], k_raw * k_cos - k_pair * k_sin, k_raw * k_cos + k_pair * k_sin).to(tl.float16)
            v = tl.load(
                v_ptr + batch * v_b_stride + head * v_h_stride + cols[:, None] * v_t_stride + offs_d[None, :] * v_d_stride,
                mask=(cols[:, None] < seq_len) & d_mask[None, :],
                other=0.,
            )
            qk = tl.dot(q, tl.trans(k)) * scale_log2e
            qk = tl.where((offs_m[:, None] < seq_len) & (cols[None, :] < seq_len), qk, -float('inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        gate = tl.load(
            gates_ptr + batch * gates_b_stride + offs_m * gates_t_stride + head * gates_h_stride,
            mask=offs_m < seq_len,
            other=-float('inf'),
        ).to(tl.float32)
        lhs = ((acc / l_i[:, None]) * tl.sigmoid(gate)[:, None]).to(tl.float16)

        for out_start in range(0, out_dim, BLOCK_O):
            offs_o = out_start + tl.arange(0, BLOCK_O)
            weight = tl.load(
                weight_ptr + offs_o[None, :] * weight_n_stride + (head * dim_head + offs_d)[:, None] * weight_k_stride,
                mask=(offs_o[None, :] < out_dim) & d_mask[:, None],
                other=0.,
            )
            contrib = tl.dot(lhs, weight)
            if HAS_BIAS:
                contrib += tl.load(bias_ptr + offs_o, mask=offs_o < out_dim, other=0.)[None, :] * (head == 0)
            tl.atomic_add(
                dst_ptr + batch * dst_b_stride + offs_m[:, None] * dst_t_stride + offs_o[None, :] * dst_n_stride,
                contrib,
                sem='relaxed',
                mask=(offs_m[:, None] < seq_len) & (offs_o[None, :] < out_dim),
            )

    @triton.jit
    def _mask_final_tanh_glu_packed_kernel(
            x_ptr,
            w_ptr,
            bias_ptr,
            out_ptr,
            total_rows: tl.constexpr,
            seq_len: tl.constexpr,
            groups: tl.constexpr,
            group_bands: tl.constexpr,
            in_dim: tl.constexpr,
            out_dim: tl.constexpr,
            x_b_stride,
            x_t_stride,
            x_g_stride,
            x_k_stride,
            w_g_stride,
            w_n_stride,
            w_k_stride,
            bias_g_stride,
            bias_n_stride,
            out_b_stride,
            out_s_stride,
            out_t_stride,
            out_n_stride,
            offset_start: tl.constexpr,
            has_bias: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        group = tl.program_id(1)
        pid_n = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = tl.arange(0, BLOCK_K)
        batch = rows // seq_len
        time = rows - batch * seq_len
        stem = group // group_bands
        band = group - stem * group_bands
        acc_a = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_b = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_start in range(0, in_dim, BLOCK_K):
            k = k_start + ks
            x = tl.load(
                x_ptr + batch[:, None] * x_b_stride + time[:, None] * x_t_stride + group * x_g_stride + k[None, :] * x_k_stride,
                mask=(rows[:, None] < total_rows) & (k[None, :] < in_dim),
                other=0.,
            )
            x = (2. * tl.sigmoid(2. * x.to(tl.float32)) - 1.).to(tl.float16)
            wa = tl.load(
                w_ptr + group * w_g_stride + cols[None, :] * w_n_stride + k[:, None] * w_k_stride,
                mask=(cols[None, :] < out_dim) & (k[:, None] < in_dim),
                other=0.,
            )
            wb = tl.load(
                w_ptr + group * w_g_stride + (out_dim + cols[None, :]) * w_n_stride + k[:, None] * w_k_stride,
                mask=(cols[None, :] < out_dim) & (k[:, None] < in_dim),
                other=0.,
            )
            acc_a += tl.dot(x, wa)
            acc_b += tl.dot(x, wb)

        if has_bias:
            acc_a += tl.load(bias_ptr + group * bias_g_stride + cols * bias_n_stride, mask=cols < out_dim, other=0.)[None, :]
            acc_b += tl.load(
                bias_ptr + group * bias_g_stride + (out_dim + cols) * bias_n_stride,
                mask=cols < out_dim,
                other=0.,
            )[None, :]

        out = acc_a * tl.sigmoid(acc_b)
        tl.store(
            out_ptr
            + batch[:, None] * out_b_stride
            + stem * out_s_stride
            + time[:, None] * out_t_stride
            + (offset_start + band * out_dim + cols[None, :]) * out_n_stride,
            out,
            mask=(rows[:, None] < total_rows) & (cols[None, :] < out_dim),
        )

    @triton.jit
    def _mask_final_tanh_glu_flat_kernel(
            x_ptr,
            w_ptr,
            bias_ptr,
            out_ptr,
            total_rows: tl.constexpr,
            seq_len: tl.constexpr,
            groups: tl.constexpr,
            in_dim: tl.constexpr,
            out_dim: tl.constexpr,
            x_b_stride,
            x_t_stride,
            x_g_stride,
            x_k_stride,
            w_g_stride,
            w_n_stride,
            w_k_stride,
            bias_g_stride,
            bias_n_stride,
            out_b_stride,
            out_t_stride,
            out_n_stride,
            offset_start: tl.constexpr,
            has_bias: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        group = tl.program_id(1)
        pid_n = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = tl.arange(0, BLOCK_K)
        batch = rows // seq_len
        time = rows - batch * seq_len
        acc_a = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_b = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_start in range(0, in_dim, BLOCK_K):
            k = k_start + ks
            x = tl.load(
                x_ptr + batch[:, None] * x_b_stride + time[:, None] * x_t_stride + group * x_g_stride + k[None, :] * x_k_stride,
                mask=(rows[:, None] < total_rows) & (k[None, :] < in_dim),
                other=0.,
            )
            x = (2. * tl.sigmoid(2. * x.to(tl.float32)) - 1.).to(tl.float16)
            wa = tl.load(
                w_ptr + group * w_g_stride + cols[None, :] * w_n_stride + k[:, None] * w_k_stride,
                mask=(cols[None, :] < out_dim) & (k[:, None] < in_dim),
                other=0.,
            )
            wb = tl.load(
                w_ptr + group * w_g_stride + (out_dim + cols[None, :]) * w_n_stride + k[:, None] * w_k_stride,
                mask=(cols[None, :] < out_dim) & (k[:, None] < in_dim),
                other=0.,
            )
            acc_a += tl.dot(x, wa)
            acc_b += tl.dot(x, wb)

        if has_bias:
            acc_a += tl.load(bias_ptr + group * bias_g_stride + cols * bias_n_stride, mask=cols < out_dim, other=0.)[None, :]
            acc_b += tl.load(
                bias_ptr + group * bias_g_stride + (out_dim + cols) * bias_n_stride,
                mask=cols < out_dim,
                other=0.,
            )[None, :]

        out = acc_a * tl.sigmoid(acc_b)
        tl.store(
            out_ptr + batch[:, None] * out_b_stride + time[:, None] * out_t_stride + (offset_start + group * out_dim + cols[None, :]) * out_n_stride,
            out,
            mask=(rows[:, None] < total_rows) & (cols[None, :] < out_dim),
        )

def _next_power_of_2(value):
    return 1 << (value - 1).bit_length()


_RTX_5090_DEVICE_CACHE = {}
_RTX_5090_PACKED64_SEQ_LENS = (938, 1723)


def _is_rtx_5090_device(device):
    if device.type != "cuda":
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    key = (device.type, index)
    cached = _RTX_5090_DEVICE_CACHE.get(key)
    if cached is not None:
        return cached
    props = torch.cuda.get_device_properties(index)
    cached = (
        (props.major, props.minor) == (12, 0)
        and props.multi_processor_count == 170
        and getattr(props, "shared_memory_per_block_optin", 0) >= 101376
    )
    _RTX_5090_DEVICE_CACHE[key] = cached
    return cached


def rms_norm_triton(x, gamma, eps=1e-12):
    if not triton_kernels_available() or not (x.is_cuda and gamma.is_cuda):
        return None
    if x.dtype != torch.float16 or gamma.dtype != x.dtype or not x.is_contiguous() or not gamma.is_contiguous():
        return None
    if x.ndim < 2 or x.shape[-1] > 1024:
        return None

    dim = x.shape[-1]
    rows = x.numel() // dim
    out = torch.empty_like(x)
    if dim in (256, 384) and rows >= 65536:
        _rms_norm_rows_kernel[(triton.cdiv(rows, 2),)](
            x,
            gamma,
            out,
            rows,
            dim,
            eps,
            BLOCK_R=2,
            BLOCK_D=_next_power_of_2(dim),
            num_warps=4,
        )
        return out
    _rms_norm_kernel[(rows,)](
        x,
        gamma,
        out,
        rows,
        dim,
        eps,
        BLOCK_D=_next_power_of_2(dim),
        num_warps=1 if dim <= 64 else 2 if dim in (256, 384) else 4,
    )
    return out


def copy_rms_norm_4d_triton(x, gamma, order, eps=1e-12):
    if not triton_kernels_available() or not (x.is_cuda and gamma.is_cuda):
        return None
    if x.dtype != torch.float16 or gamma.dtype != x.dtype or not gamma.is_contiguous() or x.ndim != 4:
        return None

    size_b, size_t, size_f, dim = x.shape
    if dim not in (256, 512) or x.stride(-1) != 1:
        return None

    if order == "bft":
        out_shape = (size_b * size_f, size_t, dim)
        order_id = 0
    elif order == "btf":
        out_shape = (size_b * size_t, size_f, dim)
        order_id = 1
    else:
        return None

    rows = out_shape[0] * out_shape[1]
    if rows < 65536:
        return None

    normed = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    residual = torch.empty_like(normed)
    _copy_rms_norm_4d_kernel[(rows,)](
        x,
        gamma,
        normed,
        residual,
        rows,
        size_b,
        size_t,
        size_f,
        dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        eps,
        order_id,
        BLOCK_D=_next_power_of_2(dim),
        num_warps=1 if dim <= 64 else 4,
    )
    return normed, residual


def rotate_qk_inplace_triton(q, k, cos, sin):
    if not triton_kernels_available() or not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda):
        return None
    if q.dtype != torch.float16 or k.dtype != q.dtype or cos.dtype != q.dtype or sin.dtype != q.dtype:
        return None
    if q.shape != k.shape or q.ndim != 4 or cos.ndim != 4 or sin.ndim != 4:
        return None

    batch, seq_len, heads, dim_head = q.shape
    if (
        dim_head % 2
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or cos.shape[1] < seq_len
        or sin.shape[1] < seq_len
        or cos.shape[-1] < dim_head
        or sin.shape[-1] < dim_head
    ):
        return None

    total_pairs = batch * seq_len * heads * (dim_head // 2)
    block_size = 128 if seq_len >= 1280 else 512 if seq_len >= 768 else 256
    _rotate_qk_inplace_kernel[(triton.cdiv(total_pairs, block_size),)](
        q,
        k,
        cos,
        sin,
        total_pairs,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        cos.stride(1),
        cos.stride(3),
        sin.stride(1),
        sin.stride(3),
        seq_len,
        heads,
        dim_head // 2,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return q, k


def attention_gate_triton(q, k, v, gates):
    if not triton_kernels_available() or not (q.is_cuda and k.is_cuda and v.is_cuda and gates.is_cuda):
        return None
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype or gates.dtype != q.dtype:
        return None
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        return None

    batch, heads, seq_len, dim_head = q.shape
    if gates.shape != (batch, seq_len, heads) or dim_head > 128:
        return None

    dst = torch.empty((batch, seq_len, heads * dim_head), device=q.device, dtype=q.dtype)
    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_M']), batch * heads)
    _attention_gate_kernel[grid](
        q,
        k,
        v,
        gates,
        dst,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        gates.stride(0),
        gates.stride(1),
        gates.stride(2),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        (dim_head ** -0.5) * 1.4426950408889634,
        seq_len,
        heads,
        dim_head,
        BLOCK_D=_next_power_of_2(dim_head),
    )
    return dst


def attention_gate_varlen_triton(q, k, v, gates):
    if not triton_kernels_available() or not (q.is_cuda and k.is_cuda and v.is_cuda and gates.is_cuda):
        return None
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype or gates.dtype != q.dtype:
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    batch, heads, q_len, dim_head = q.shape
    if k.shape[:2] != (batch, heads) or v.shape != k.shape or k.shape[-1] != dim_head:
        return None
    kv_len = k.shape[2]
    if gates.shape != (batch, q_len, heads) or dim_head > 128 or kv_len < 1:
        return None

    dst = torch.empty((batch, q_len, heads * dim_head), device=q.device, dtype=q.dtype)
    grid = lambda meta: (triton.cdiv(q_len, meta['BLOCK_M']), batch * heads)
    _attention_gate_varlen_kernel[grid](
        q,
        k,
        v,
        gates,
        dst,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        gates.stride(0),
        gates.stride(1),
        gates.stride(2),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        (dim_head ** -0.5) * 1.4426950408889634,
        q_len,
        kv_len,
        heads,
        dim_head,
        BLOCK_D=_next_power_of_2(dim_head),
    )
    return dst


def _is_packed_qkv64(q, k, v, gates, batch, heads, seq_len):
    if q.shape[-1] != 64 or q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    if q.stride(0) != k.stride(0) or q.stride(0) != v.stride(0) or q.stride(2) != k.stride(2) or q.stride(2) != v.stride(2):
        return False
    packed_t_stride = 3 * heads * 64
    return (
        q.stride(0) == seq_len * packed_t_stride
        and q.stride(2) == packed_t_stride
        and q.stride(1) == 64
        and k.stride(1) == 64
        and v.stride(1) == 64
        and k.data_ptr() - q.data_ptr() == heads * 64 * q.element_size()
        and v.data_ptr() - q.data_ptr() == 2 * heads * 64 * q.element_size()
        and gates.is_contiguous()
        and gates.shape == (batch, seq_len, heads)
    )


def attention_gate_packed64_5090_triton(q, k, v, gates):
    if not triton_kernels_available() or not (q.is_cuda and k.is_cuda and v.is_cuda and gates.is_cuda):
        return None
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype or gates.dtype != q.dtype:
        return None
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        return None

    batch, heads, seq_len, dim_head = q.shape
    if (
        dim_head != 64
        or seq_len not in _RTX_5090_PACKED64_SEQ_LENS
        or not _is_rtx_5090_device(q.device)
        or not _is_packed_qkv64(q, k, v, gates, batch, heads, seq_len)
    ):
        return None

    dst = torch.empty((batch, seq_len, heads * 64), device=q.device, dtype=q.dtype)
    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_M']), batch * heads)
    _attention_gate_packed64_5090_kernel[grid](
        q,
        k,
        v,
        gates,
        dst,
        (64 ** -0.5) * 1.4426950408889634,
        seq_len,
        heads,
    )
    return dst


def attention_gate_out_triton(q, k, v, gates, weight, bias, residual=None):
    if not triton_kernels_available() or not (q.is_cuda and k.is_cuda and v.is_cuda and gates.is_cuda and weight.is_cuda):
        return None
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype or gates.dtype != q.dtype or weight.dtype != q.dtype:
        return None
    if bias is not None and (not bias.is_cuda or bias.dtype != q.dtype):
        return None
    if residual is not None and (not residual.is_cuda or residual.dtype != q.dtype):
        return None
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or weight.ndim != 2:
        return None

    batch, heads, seq_len, dim_head = q.shape
    out_dim, in_dim = weight.shape
    if gates.shape != (batch, seq_len, heads) or in_dim != heads * dim_head or dim_head > 128:
        return None
    if bias is not None and bias.shape != (out_dim,):
        return None
    if residual is not None and residual.shape != (batch, seq_len, out_dim):
        return None

    dst = torch.empty((batch, seq_len, out_dim), device=q.device, dtype=q.dtype)
    if bias is None and residual is None:
        dst.zero_()
    else:
        total = dst.numel()
        _init_linear_out_kernel[(triton.cdiv(total, 256),)](
            bias if bias is not None else weight,
            residual if residual is not None else dst,
            dst,
            total,
            seq_len,
            out_dim,
            0 if residual is None else residual.stride(0),
            0 if residual is None else residual.stride(1),
            0 if residual is None else residual.stride(2),
            HAS_BIAS=bias is not None,
            HAS_RESIDUAL=residual is not None,
            BLOCK_SIZE=256,
            num_warps=4,
        )

    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_M']), batch * heads)
    _attention_gate_out_kernel[grid](
        q,
        k,
        v,
        gates,
        weight,
        dst,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        gates.stride(0),
        gates.stride(1),
        gates.stride(2),
        weight.stride(0),
        weight.stride(1),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        (dim_head ** -0.5) * 1.4426950408889634,
        seq_len,
        heads,
        dim_head,
        out_dim,
        BLOCK_D=_next_power_of_2(dim_head),
    )
    return dst


def attention_gate_out_inplace_triton(q, k, v, gates, weight, residual, bias=None):
    if not triton_kernels_available() or not (q.is_cuda and k.is_cuda and v.is_cuda and gates.is_cuda and weight.is_cuda and residual.is_cuda):
        return None
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype or gates.dtype != q.dtype or weight.dtype != q.dtype or residual.dtype != q.dtype:
        return None
    if bias is not None and (not bias.is_cuda or bias.dtype != q.dtype):
        return None
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or weight.ndim != 2:
        return None

    batch, heads, seq_len, dim_head = q.shape
    out_dim, in_dim = weight.shape
    if gates.shape != (batch, seq_len, heads) or in_dim != heads * dim_head or residual.shape != (batch, seq_len, out_dim) or dim_head > 128:
        return None
    if bias is not None:
        if bias.shape != (out_dim,):
            return None
        total = residual.numel()
        _init_linear_out_kernel[(triton.cdiv(total, 256),)](
            bias,
            residual,
            residual,
            total,
            seq_len,
            out_dim,
            residual.stride(0),
            residual.stride(1),
            residual.stride(2),
            HAS_BIAS=True,
            HAS_RESIDUAL=True,
            BLOCK_SIZE=256,
            num_warps=4,
        )

    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_M']), batch * heads)
    _attention_gate_out_kernel[grid](
        q,
        k,
        v,
        gates,
        weight,
        residual,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        gates.stride(0),
        gates.stride(1),
        gates.stride(2),
        weight.stride(0),
        weight.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        (dim_head ** -0.5) * 1.4426950408889634,
        seq_len,
        heads,
        dim_head,
        out_dim,
        BLOCK_D=_next_power_of_2(dim_head),
    )
    return residual


def attention_gate_out_rope_inplace_triton(q, k, v, gates, weight, residual, cos, sin, bias=None):
    if not triton_kernels_available() or not (q.is_cuda and k.is_cuda and v.is_cuda and gates.is_cuda and weight.is_cuda and residual.is_cuda and cos.is_cuda and sin.is_cuda):
        return None
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype or gates.dtype != q.dtype or weight.dtype != q.dtype or residual.dtype != q.dtype or cos.dtype != q.dtype or sin.dtype != q.dtype:
        return None
    if bias is not None and (not bias.is_cuda or bias.dtype != q.dtype):
        return None
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or weight.ndim != 2 or cos.ndim != 4 or sin.ndim != 4:
        return None

    batch, heads, seq_len, dim_head = q.shape
    out_dim, in_dim = weight.shape
    if (
        seq_len > 128
        or dim_head > 128
        or dim_head % 2
        or gates.shape != (batch, seq_len, heads)
        or in_dim != heads * dim_head
        or residual.shape != (batch, seq_len, out_dim)
        or cos.shape[1] < seq_len
        or sin.shape[1] < seq_len
        or cos.shape[-1] < dim_head
        or sin.shape[-1] < dim_head
    ):
        return None
    if bias is not None and bias.shape != (out_dim,):
        return None

    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_M']), batch * heads)
    _attention_gate_out_rope_kernel[grid](
        q,
        k,
        v,
        gates,
        weight,
        cos,
        sin,
        bias if bias is not None else weight,
        residual,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        gates.stride(0),
        gates.stride(1),
        gates.stride(2),
        weight.stride(0),
        weight.stride(1),
        cos.stride(1),
        cos.stride(3),
        sin.stride(1),
        sin.stride(3),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        (dim_head ** -0.5) * 1.4426950408889634,
        seq_len,
        heads,
        dim_head,
        out_dim,
        HAS_BIAS=bias is not None,
        BLOCK_D=_next_power_of_2(dim_head),
    )
    return residual


def mask_final_tanh_glu_packed_triton(x, weight, bias, result, group_bands, offset_start, out_dim):
    if not triton_kernels_available() or not (x.is_cuda and weight.is_cuda and result.is_cuda):
        return False
    if x.dtype != torch.float16 or weight.dtype != x.dtype or result.dtype != x.dtype:
        return False
    if bias is not None and (not bias.is_cuda or bias.dtype != x.dtype):
        return False
    if x.ndim != 4 or weight.ndim != 3 or result.ndim != 4:
        return False

    batch, seq_len, groups, in_dim = x.shape
    stem_count = groups // group_bands
    if (
        group_bands <= 0
        or groups != weight.shape[0]
        or groups != stem_count * group_bands
        or weight.shape[1] != out_dim * 2
        or weight.shape[2] != in_dim
        or result.shape[0] != batch
        or result.shape[1] != stem_count
        or result.shape[2] != seq_len
        or out_dim > 96
    ):
        return False
    if bias is not None and bias.shape != (groups, out_dim * 2):
        return False

    block_m = 32 if out_dim <= 32 else 64
    block_n = 16 if out_dim <= 16 else 32
    block_k = 64
    grid = (triton.cdiv(batch * seq_len, block_m), groups, triton.cdiv(out_dim, block_n))
    _mask_final_tanh_glu_packed_kernel[grid](
        x,
        weight,
        bias if bias is not None else weight,
        result,
        batch * seq_len,
        seq_len,
        groups,
        group_bands,
        in_dim,
        out_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        0 if bias is None else bias.stride(0),
        0 if bias is None else bias.stride(1),
        result.stride(0),
        result.stride(1),
        result.stride(2),
        result.stride(3),
        offset_start,
        bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return True


def mask_final_tanh_glu_flat_triton(x, weight, bias, result, offset_start, out_dim):
    if not triton_kernels_available() or not (x.is_cuda and weight.is_cuda and result.is_cuda):
        return False
    if x.dtype != torch.float16 or weight.dtype != x.dtype or result.dtype != x.dtype:
        return False
    if bias is not None and (not bias.is_cuda or bias.dtype != x.dtype):
        return False
    if x.ndim != 4 or weight.ndim != 3 or result.ndim != 3:
        return False

    batch, seq_len, groups, in_dim = x.shape
    if (
        groups != weight.shape[0]
        or weight.shape[1] != out_dim * 2
        or weight.shape[2] != in_dim
        or result.shape[0] != batch
        or result.shape[1] != seq_len
        or out_dim > 96
    ):
        return False
    if bias is not None and bias.shape != (groups, out_dim * 2):
        return False

    block_m = 32 if out_dim <= 32 else 64
    block_n = 16 if out_dim <= 16 else 32
    block_k = 64
    grid = (triton.cdiv(batch * seq_len, block_m), groups, triton.cdiv(out_dim, block_n))
    _mask_final_tanh_glu_flat_kernel[grid](
        x,
        weight,
        bias if bias is not None else weight,
        result,
        batch * seq_len,
        seq_len,
        groups,
        in_dim,
        out_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        0 if bias is None else bias.stride(0),
        0 if bias is None else bias.stride(1),
        result.stride(0),
        result.stride(1),
        result.stride(2),
        offset_start,
        bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return True
