#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

template <typename scalar_t>
__global__ void band_split_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ normed,
    float* __restrict__ inv_norm,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features,
    int64_t x_stride_m,
    int64_t x_stride_g,
    int64_t x_stride_i) {
  extern __shared__ float shared[];
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;

  float sum_sq = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float value = static_cast<float>(x[m * x_stride_m + g * x_stride_g + i * x_stride_i]);
    sum_sq += value * value;
  }
  shared[tid] = sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float scale = sqrtf(static_cast<float>(in_features));
  const float norm = fmaxf(sqrtf(shared[0]), 1.0e-12f);
  const float inv = 1.0f / norm;
  if (tid == 0) {
    inv_norm[m * group_count + g] = inv;
  }

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float value = static_cast<float>(x[m * x_stride_m + g * x_stride_g + i * x_stride_i]);
    normed[(m * group_count + g) * in_features + i] = static_cast<scalar_t>(value * inv * scale);
  }
  __syncthreads();

  for (int64_t o = tid; o < out_features; o += blockDim.x) {
    float acc = bias == nullptr ? 0.0f : static_cast<float>(bias[g * out_features + o]);
    for (int64_t i = 0; i < in_features; ++i) {
      const float value = static_cast<float>(x[m * x_stride_m + g * x_stride_g + i * x_stride_i]);
      const float feature = value * inv * scale * static_cast<float>(gamma[g * in_features + i]);
      acc += feature * static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    out[(m * group_count + g) * out_features + o] = static_cast<scalar_t>(acc);
  }
}

std::vector<at::Tensor> band_split_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  const c10::cuda::CUDAGuard device_guard(x.device());

  const auto m_count = x.size(0);
  const auto group_count = x.size(1);
  const auto in_features = x.size(2);
  const auto out_features = weight.size(1);

  auto out = at::empty({m_count, group_count, out_features}, x.options());
  auto normed = at::empty({m_count, group_count, in_features}, x.options());
  auto inv_norm = at::empty({m_count, group_count}, x.options().dtype(at::kFloat));

  const int threads = 256;
  const dim3 blocks(m_count, group_count);
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "band_split_forward_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        band_split_forward_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
            x.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_ptr,
            out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            m_count,
            group_count,
            in_features,
            out_features,
            x.stride(0),
            x.stride(1),
            x.stride(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, normed, inv_norm};
}

template <typename scalar_t>
__global__ void band_split_backward_input_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_x,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  extern __shared__ float shared[];
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(in_features));
  const float inv = inv_norm[m * group_count + g];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                     static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[g * in_features + i]);
    dot_partial += grad_normed * static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  shared[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                     static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[g * in_features + i]);
    const float normed_value = static_cast<float>(normed[(m * group_count + g) * in_features + i]);
    const float value = inv * (grad_normed * scale - normed_value * (dot / scale));
    grad_x[(m * group_count + g) * in_features + i] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void band_split_backward_weight_bias_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const scalar_t* __restrict__ gamma,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  const int64_t g = blockIdx.x;
  const int64_t o = blockIdx.y;
  const int tid = threadIdx.x;

  if (grad_bias != nullptr && tid == 0) {
    float bias_acc = 0.0f;
    for (int64_t m = 0; m < m_count; ++m) {
      bias_acc += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]);
    }
    grad_bias[g * out_features + o] = static_cast<scalar_t>(bias_acc);
  }

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float weight_acc = 0.0f;
    const float gamma_value = static_cast<float>(gamma[g * in_features + i]);
    for (int64_t m = 0; m < m_count; ++m) {
      weight_acc += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                    static_cast<float>(normed[(m * group_count + g) * in_features + i]) * gamma_value;
    }
    grad_weight[(g * out_features + o) * in_features + i] = static_cast<scalar_t>(weight_acc);
  }
}

template <typename scalar_t>
__global__ void band_split_backward_gamma_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_gamma,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  const int64_t g = blockIdx.x;
  const int64_t i = blockIdx.y;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t index = tid; index < m_count * out_features; index += blockDim.x) {
    const int64_t m = index / out_features;
    const int64_t o = index % out_features;
    acc += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
           static_cast<float>(weight[(g * out_features + o) * in_features + i]) *
           static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  extern __shared__ float shared[];
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[g * in_features + i] = static_cast<scalar_t>(shared[0]);
  }
}

std::vector<at::Tensor> band_split_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    bool has_bias) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());

  const auto m_count = grad_out.size(0);
  const auto group_count = grad_out.size(1);
  const auto out_features = grad_out.size(2);
  const auto in_features = normed.size(2);

  auto grad_x = at::empty_like(normed);
  auto grad_gamma = at::empty_like(gamma);
  auto grad_weight = at::empty_like(weight);
  at::Tensor grad_bias = has_bias ? at::empty({group_count, out_features}, grad_out.options()) : at::Tensor();

  const int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "band_split_backward_cuda",
      [&] {
        band_split_backward_input_kernel<scalar_t><<<dim3(m_count, group_count), threads, shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_weight_bias_kernel<scalar_t><<<dim3(group_count, out_features), threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            has_bias ? grad_bias.data_ptr<scalar_t>() : nullptr,
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_gamma_kernel<scalar_t><<<dim3(group_count, in_features), threads, shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_gamma.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma, grad_weight, grad_bias};
}

template <typename scalar_t>
__global__ void band_split_backward_input_v2_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_x,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* grad_normed_cache = shared + blockDim.x;
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(in_features));
  const float inv = inv_norm[m * group_count + g];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                     static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[g * in_features + i]);
    grad_normed_cache[i] = grad_normed;
    dot_partial += grad_normed * static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  reduction[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  const float dot = reduction[0];

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float normed_value = static_cast<float>(normed[(m * group_count + g) * in_features + i]);
    const float value = inv * (grad_normed_cache[i] * scale - normed_value * (dot / scale));
    grad_x[(m * group_count + g) * in_features + i] = static_cast<scalar_t>(value);
  }
}

std::vector<at::Tensor> band_split_backward_v2_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    bool has_bias) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());

  const auto m_count = grad_out.size(0);
  const auto group_count = grad_out.size(1);
  const auto out_features = grad_out.size(2);
  const auto in_features = normed.size(2);

  const int threads = 256;
  const size_t input_shared_bytes = (threads + static_cast<size_t>(in_features)) * sizeof(float);
  if (input_shared_bytes > 48 * 1024) {
    return band_split_backward_cuda(grad_out, normed, inv_norm, gamma, weight, has_bias);
  }

  auto grad_x = at::empty_like(normed);
  auto grad_gamma = at::empty_like(gamma);
  auto grad_weight = at::empty_like(weight);
  at::Tensor grad_bias = has_bias ? at::empty({group_count, out_features}, grad_out.options()) : at::Tensor();

  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "band_split_backward_v2_cuda",
      [&] {
        band_split_backward_input_v2_kernel<scalar_t><<<dim3(m_count, group_count), threads, input_shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_weight_bias_kernel<scalar_t><<<dim3(group_count, out_features), threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            has_bias ? grad_bias.data_ptr<scalar_t>() : nullptr,
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_gamma_kernel<scalar_t><<<dim3(group_count, in_features), threads, shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_gamma.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma, grad_weight, grad_bias};
}

template <typename scalar_t>
__global__ void band_split_backward_input_v3_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_x,
    float* __restrict__ grad_scaled_cache,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* grad_normed_cache = shared + blockDim.x;
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(in_features));
  const float inv = inv_norm[m * group_count + g];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                     static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[g * in_features + i]);
    grad_scaled_cache[(m * group_count + g) * in_features + i] = grad_scaled;
    grad_normed_cache[i] = grad_normed;
    dot_partial += grad_normed * static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  reduction[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  const float dot = reduction[0];

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float normed_value = static_cast<float>(normed[(m * group_count + g) * in_features + i]);
    const float value = inv * (grad_normed_cache[i] * scale - normed_value * (dot / scale));
    grad_x[(m * group_count + g) * in_features + i] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void band_split_backward_gamma_from_scaled_kernel(
    const float* __restrict__ grad_scaled_cache,
    const scalar_t* __restrict__ normed,
    scalar_t* __restrict__ grad_gamma,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features) {
  const int64_t g = blockIdx.x;
  const int64_t i = blockIdx.y;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t m = tid; m < m_count; m += blockDim.x) {
    acc += grad_scaled_cache[(m * group_count + g) * in_features + i] *
           static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  extern __shared__ float shared[];
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[g * in_features + i] = static_cast<scalar_t>(shared[0]);
  }
}

std::vector<at::Tensor> band_split_backward_v3_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    bool has_bias) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());

  const auto m_count = grad_out.size(0);
  const auto group_count = grad_out.size(1);
  const auto out_features = grad_out.size(2);
  const auto in_features = normed.size(2);

  const int threads = 256;
  const size_t input_shared_bytes = (threads + static_cast<size_t>(in_features)) * sizeof(float);
  if (input_shared_bytes > 48 * 1024) {
    return band_split_backward_v2_cuda(grad_out, normed, inv_norm, gamma, weight, has_bias);
  }

  auto grad_x = at::empty_like(normed);
  auto grad_gamma = at::empty_like(gamma);
  auto grad_weight = at::empty_like(weight);
  auto grad_scaled_cache = at::empty({m_count, group_count, in_features}, grad_out.options().dtype(at::kFloat));
  at::Tensor grad_bias = has_bias ? at::empty({group_count, out_features}, grad_out.options()) : at::Tensor();

  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "band_split_backward_v3_cuda",
      [&] {
        band_split_backward_input_v3_kernel<scalar_t><<<dim3(m_count, group_count), threads, input_shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_scaled_cache.data_ptr<float>(),
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_weight_bias_kernel<scalar_t><<<dim3(group_count, out_features), threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            has_bias ? grad_bias.data_ptr<scalar_t>() : nullptr,
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_gamma_from_scaled_kernel<scalar_t>
            <<<dim3(group_count, in_features), threads, shared_bytes, stream>>>(
                grad_scaled_cache.data_ptr<float>(),
                normed.data_ptr<scalar_t>(),
                grad_gamma.data_ptr<scalar_t>(),
                m_count,
                group_count,
                in_features);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma, grad_weight, grad_bias};
}

template <typename scalar_t>
__global__ void band_split_ragged_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* const* __restrict__ gammas,
    const scalar_t* const* __restrict__ weights,
    const scalar_t* const* __restrict__ biases,
    const int64_t* __restrict__ offsets,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ normed,
    float* __restrict__ inv_norm,
    int64_t m_count,
    int64_t band_count,
    int64_t out_features,
    int64_t total_in) {
  extern __shared__ float shared[];
  const int64_t m = blockIdx.x;
  const int64_t band = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t in_features = end - start;

  float sum_sq = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float value = static_cast<float>(x[m * total_in + start + i]);
    sum_sq += value * value;
  }
  shared[tid] = sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float scale = sqrtf(static_cast<float>(in_features));
  const float norm = fmaxf(sqrtf(shared[0]), 1.0e-12f);
  const float inv = 1.0f / norm;
  if (tid == 0) {
    inv_norm[m * band_count + band] = inv;
  }

  const scalar_t* gamma = gammas[band];
  const scalar_t* weight = weights[band];
  const scalar_t* bias = biases[band];

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float value = static_cast<float>(x[m * total_in + start + i]);
    normed[m * total_in + start + i] = static_cast<scalar_t>(value * inv * scale);
  }
  __syncthreads();

  for (int64_t o = tid; o < out_features; o += blockDim.x) {
    float acc = static_cast<float>(bias[o]);
    for (int64_t i = 0; i < in_features; ++i) {
      const float value = static_cast<float>(x[m * total_in + start + i]);
      const float feature = value * inv * scale * static_cast<float>(gamma[i]);
      acc += feature * static_cast<float>(weight[o * in_features + i]);
    }
    out[(m * band_count + band) * out_features + o] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
at::Tensor tensor_from_host_ptrs(const std::vector<const scalar_t*>& ptrs, const at::TensorOptions& options) {
  auto tensor = at::empty({static_cast<int64_t>(ptrs.size())}, options.dtype(at::kLong));
  auto cpu = at::empty(
      {static_cast<int64_t>(ptrs.size())},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
  auto* values = cpu.data_ptr<int64_t>();
  for (size_t index = 0; index < ptrs.size(); ++index) {
    values[index] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptrs[index]));
  }
  tensor.copy_(cpu, true);
  return tensor;
}

at::Tensor contiguous_if_needed(const at::Tensor& tensor) {
  return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

std::vector<at::Tensor> band_split_ragged_forward_cuda(
    const at::Tensor& x,
    const std::vector<at::Tensor>& gammas,
    const std::vector<at::Tensor>& weights,
    const std::vector<at::Tensor>& biases,
    const std::vector<int64_t>& dim_inputs,
    const at::Tensor& dim_offsets) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contiguous = x.contiguous();

  const auto m_count = x_contiguous.size(0);
  const auto total_in = x_contiguous.size(1);
  const auto band_count = static_cast<int64_t>(weights.size());
  const auto out_features = weights[0].size(0);
  auto out = at::empty({m_count, band_count, out_features}, x_contiguous.options());
  auto normed = at::empty_like(x_contiguous);
  auto inv_norm = at::empty({m_count, band_count}, x_contiguous.options().dtype(at::kFloat));

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "band_split_ragged_forward_cuda",
      [&] {
        std::vector<at::Tensor> gamma_contiguous;
        std::vector<at::Tensor> weight_contiguous;
        std::vector<at::Tensor> bias_contiguous;
        std::vector<const scalar_t*> gamma_ptrs;
        std::vector<const scalar_t*> weight_ptrs;
        std::vector<const scalar_t*> bias_ptrs;
        gamma_contiguous.reserve(gammas.size());
        weight_contiguous.reserve(weights.size());
        bias_contiguous.reserve(biases.size());
        gamma_ptrs.reserve(gammas.size());
        weight_ptrs.reserve(weights.size());
        bias_ptrs.reserve(biases.size());
        for (size_t index = 0; index < weights.size(); ++index) {
          gamma_contiguous.push_back(contiguous_if_needed(gammas[index]));
          weight_contiguous.push_back(contiguous_if_needed(weights[index]));
          bias_contiguous.push_back(contiguous_if_needed(biases[index]));
          gamma_ptrs.push_back(gamma_contiguous.back().data_ptr<scalar_t>());
          weight_ptrs.push_back(weight_contiguous.back().data_ptr<scalar_t>());
          bias_ptrs.push_back(bias_contiguous.back().data_ptr<scalar_t>());
        }
        auto gamma_ptr_tensor = tensor_from_host_ptrs(gamma_ptrs, x_contiguous.options());
        auto weight_ptr_tensor = tensor_from_host_ptrs(weight_ptrs, x_contiguous.options());
        auto bias_ptr_tensor = tensor_from_host_ptrs(bias_ptrs, x_contiguous.options());
        band_split_ragged_forward_kernel<scalar_t><<<dim3(m_count, band_count), threads, shared_bytes, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t* const*>(gamma_ptr_tensor.data_ptr<int64_t>()),
            reinterpret_cast<const scalar_t* const*>(weight_ptr_tensor.data_ptr<int64_t>()),
            reinterpret_cast<const scalar_t* const*>(bias_ptr_tensor.data_ptr<int64_t>()),
            dim_offsets.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            m_count,
            band_count,
            out_features,
            total_in);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, normed, inv_norm};
}

template <typename scalar_t>
__global__ void band_split_ragged_backward_input_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const scalar_t* const* __restrict__ gammas,
    const scalar_t* const* __restrict__ weights,
    const int64_t* __restrict__ offsets,
    scalar_t* __restrict__ grad_x,
    float* __restrict__ grad_scaled_cache,
    int64_t m_count,
    int64_t band_count,
    int64_t out_features,
    int64_t total_in) {
  extern __shared__ float shared[];
  const int64_t m = blockIdx.x;
  const int64_t band = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t in_features = end - start;
  const float scale = sqrtf(static_cast<float>(in_features));
  const float inv = inv_norm[m * band_count + band];
  const scalar_t* gamma = gammas[band];
  const scalar_t* weight = weights[band];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * band_count + band) * out_features + o]) *
                     static_cast<float>(weight[o * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[i]);
    grad_scaled_cache[m * total_in + start + i] = grad_scaled;
    dot_partial += grad_normed * static_cast<float>(normed[m * total_in + start + i]);
  }

  shared[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * band_count + band) * out_features + o]) *
                     static_cast<float>(weight[o * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[i]);
    const float normed_value = static_cast<float>(normed[m * total_in + start + i]);
    const float value = inv * (grad_normed * scale - normed_value * (dot / scale));
    grad_x[m * total_in + start + i] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void band_split_ragged_backward_weight_bias_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const scalar_t* const* __restrict__ gammas,
    scalar_t* const* __restrict__ grad_weights,
    scalar_t* const* __restrict__ grad_biases,
    const int64_t* __restrict__ offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t out_features,
    int64_t total_in) {
  const int64_t band = blockIdx.x;
  const int64_t o = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t in_features = end - start;
  const scalar_t* gamma = gammas[band];
  scalar_t* grad_weight = grad_weights[band];
  scalar_t* grad_bias = grad_biases[band];

  if (tid == 0) {
    float bias_acc = 0.0f;
    for (int64_t m = 0; m < m_count; ++m) {
      bias_acc += static_cast<float>(grad_out[(m * band_count + band) * out_features + o]);
    }
    grad_bias[o] = static_cast<scalar_t>(bias_acc);
  }

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float weight_acc = 0.0f;
    const float gamma_value = static_cast<float>(gamma[i]);
    for (int64_t m = 0; m < m_count; ++m) {
      weight_acc += static_cast<float>(grad_out[(m * band_count + band) * out_features + o]) *
                    static_cast<float>(normed[m * total_in + start + i]) * gamma_value;
    }
    grad_weight[o * in_features + i] = static_cast<scalar_t>(weight_acc);
  }
}

template <typename scalar_t>
__global__ void band_split_ragged_backward_gamma_from_scaled_kernel(
    const float* __restrict__ grad_scaled_cache,
    const scalar_t* __restrict__ normed,
    scalar_t* const* __restrict__ grad_gammas,
    const int64_t* __restrict__ offsets,
    int64_t m_count,
    int64_t max_in_features,
    int64_t total_in) {
  const int64_t band = blockIdx.x;
  const int64_t i = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t in_features = end - start;
  if (i >= in_features || i >= max_in_features) {
    return;
  }

  float acc = 0.0f;
  for (int64_t m = tid; m < m_count; m += blockDim.x) {
    acc += grad_scaled_cache[m * total_in + start + i] * static_cast<float>(normed[m * total_in + start + i]);
  }

  extern __shared__ float shared[];
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gammas[band][i] = static_cast<scalar_t>(shared[0]);
  }
}

std::vector<at::Tensor> band_split_ragged_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const std::vector<at::Tensor>& gammas,
    const std::vector<at::Tensor>& weights,
    const std::vector<int64_t>& dim_inputs,
    const at::Tensor& dim_offsets) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto normed_contiguous = normed.contiguous();

  const auto m_count = grad_out_contiguous.size(0);
  const auto band_count = grad_out_contiguous.size(1);
  const auto out_features = grad_out_contiguous.size(2);
  const auto total_in = normed_contiguous.size(1);

  auto grad_x = at::empty_like(normed_contiguous);
  auto grad_scaled_cache = at::empty({m_count, total_in}, grad_out_contiguous.options().dtype(at::kFloat));
  std::vector<at::Tensor> grad_gammas;
  std::vector<at::Tensor> grad_weights;
  std::vector<at::Tensor> grad_biases;
  grad_gammas.reserve(gammas.size());
  grad_weights.reserve(weights.size());
  grad_biases.reserve(weights.size());
  for (size_t index = 0; index < weights.size(); ++index) {
    grad_gammas.push_back(at::empty_like(gammas[index]));
    grad_weights.push_back(at::empty_like(weights[index]));
    grad_biases.push_back(at::empty({out_features}, grad_out_contiguous.options()));
  }

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto max_in_features = *std::max_element(dim_inputs.begin(), dim_inputs.end());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "band_split_ragged_backward_cuda",
      [&] {
        std::vector<at::Tensor> gamma_contiguous;
        std::vector<at::Tensor> weight_contiguous;
        std::vector<const scalar_t*> gamma_ptrs;
        std::vector<const scalar_t*> weight_ptrs;
        std::vector<const scalar_t*> grad_gamma_ptrs;
        std::vector<const scalar_t*> grad_weight_ptrs;
        std::vector<const scalar_t*> grad_bias_ptrs;
        gamma_contiguous.reserve(gammas.size());
        weight_contiguous.reserve(weights.size());
        gamma_ptrs.reserve(gammas.size());
        weight_ptrs.reserve(weights.size());
        grad_gamma_ptrs.reserve(gammas.size());
        grad_weight_ptrs.reserve(weights.size());
        grad_bias_ptrs.reserve(weights.size());
        for (size_t index = 0; index < weights.size(); ++index) {
          gamma_contiguous.push_back(contiguous_if_needed(gammas[index]));
          weight_contiguous.push_back(contiguous_if_needed(weights[index]));
          gamma_ptrs.push_back(gamma_contiguous.back().data_ptr<scalar_t>());
          weight_ptrs.push_back(weight_contiguous.back().data_ptr<scalar_t>());
          grad_gamma_ptrs.push_back(grad_gammas[index].data_ptr<scalar_t>());
          grad_weight_ptrs.push_back(grad_weights[index].data_ptr<scalar_t>());
          grad_bias_ptrs.push_back(grad_biases[index].data_ptr<scalar_t>());
        }
        auto gamma_ptr_tensor = tensor_from_host_ptrs(gamma_ptrs, grad_out_contiguous.options());
        auto weight_ptr_tensor = tensor_from_host_ptrs(weight_ptrs, grad_out_contiguous.options());
        auto grad_gamma_ptr_tensor = tensor_from_host_ptrs(grad_gamma_ptrs, grad_out_contiguous.options());
        auto grad_weight_ptr_tensor = tensor_from_host_ptrs(grad_weight_ptrs, grad_out_contiguous.options());
        auto grad_bias_ptr_tensor = tensor_from_host_ptrs(grad_bias_ptrs, grad_out_contiguous.options());

        band_split_ragged_backward_input_kernel<scalar_t>
            <<<dim3(m_count, band_count), threads, shared_bytes, stream>>>(
                grad_out_contiguous.data_ptr<scalar_t>(),
                normed_contiguous.data_ptr<scalar_t>(),
                inv_norm.data_ptr<float>(),
                reinterpret_cast<const scalar_t* const*>(gamma_ptr_tensor.data_ptr<int64_t>()),
                reinterpret_cast<const scalar_t* const*>(weight_ptr_tensor.data_ptr<int64_t>()),
                dim_offsets.data_ptr<int64_t>(),
                grad_x.data_ptr<scalar_t>(),
                grad_scaled_cache.data_ptr<float>(),
                m_count,
                band_count,
                out_features,
                total_in);
        band_split_ragged_backward_weight_bias_kernel<scalar_t>
            <<<dim3(band_count, out_features), threads, 0, stream>>>(
                grad_out_contiguous.data_ptr<scalar_t>(),
                normed_contiguous.data_ptr<scalar_t>(),
                reinterpret_cast<const scalar_t* const*>(gamma_ptr_tensor.data_ptr<int64_t>()),
                reinterpret_cast<scalar_t* const*>(grad_weight_ptr_tensor.data_ptr<int64_t>()),
                reinterpret_cast<scalar_t* const*>(grad_bias_ptr_tensor.data_ptr<int64_t>()),
                dim_offsets.data_ptr<int64_t>(),
                m_count,
                band_count,
                out_features,
                total_in);
        band_split_ragged_backward_gamma_from_scaled_kernel<scalar_t>
            <<<dim3(band_count, max_in_features), threads, shared_bytes, stream>>>(
                grad_scaled_cache.data_ptr<float>(),
                normed_contiguous.data_ptr<scalar_t>(),
                reinterpret_cast<scalar_t* const*>(grad_gamma_ptr_tensor.data_ptr<int64_t>()),
                dim_offsets.data_ptr<int64_t>(),
                m_count,
                max_in_features,
                total_in);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<at::Tensor> result;
  result.reserve(1 + grad_gammas.size() + grad_weights.size() + grad_biases.size());
  result.push_back(grad_x);
  for (auto& grad_gamma : grad_gammas) {
    result.push_back(grad_gamma);
  }
  for (auto& grad_weight : grad_weights) {
    result.push_back(grad_weight);
  }
  for (auto& grad_bias : grad_biases) {
    result.push_back(grad_bias);
  }
  return result;
}

template <typename scalar_t>
__global__ void grouped_linear_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features,
    int64_t x_stride_m,
    int64_t x_stride_g,
    int64_t x_stride_i) {
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;

  for (int64_t o = tid; o < out_features; o += blockDim.x) {
    float acc = bias == nullptr ? 0.0f : static_cast<float>(bias[g * out_features + o]);
    for (int64_t i = 0; i < in_features; ++i) {
      acc += static_cast<float>(x[m * x_stride_m + g * x_stride_g + i * x_stride_i]) *
             static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    out[(m * group_count + g) * out_features + o] = static_cast<scalar_t>(acc);
  }
}

at::Tensor grouped_linear_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  const c10::cuda::CUDAGuard device_guard(x.device());

  const auto m_count = x.size(0);
  const auto group_count = x.size(1);
  const auto in_features = x.size(2);
  const auto out_features = weight.size(1);

  auto out = at::empty({m_count, group_count, out_features}, x.options());
  const int threads = 256;
  const dim3 blocks(m_count, group_count);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "grouped_linear_forward_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        grouped_linear_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_ptr,
            out.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features,
            x.stride(0),
            x.stride(1),
            x.stride(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}

template <typename scalar_t>
__global__ void rotary_forward_kernel(
    const scalar_t* __restrict__ t,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t seq_len,
    int64_t heads,
    int64_t dim) {
  const int64_t pairs = dim / 2;
  const int64_t total = batch * seq_len * heads * pairs;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    const int64_t pair = index % pairs;
    const int64_t head = (index / pairs) % heads;
    const int64_t seq = (index / (pairs * heads)) % seq_len;
    const int64_t batch_index = index / (pairs * heads * seq_len);
    const int64_t base = ((batch_index * seq_len + seq) * heads + head) * dim + pair * 2;
    const float c = static_cast<float>(cos[seq * pairs + pair]);
    const float s = static_cast<float>(sin[seq * pairs + pair]);
    const float even = static_cast<float>(t[base]);
    const float odd = static_cast<float>(t[base + 1]);
    out[base] = static_cast<scalar_t>(even * c - odd * s);
    out[base + 1] = static_cast<scalar_t>(odd * c + even * s);
  }
}

template <typename scalar_t>
__global__ void rotary_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ grad_t,
    int64_t batch,
    int64_t seq_len,
    int64_t heads,
    int64_t dim) {
  const int64_t pairs = dim / 2;
  const int64_t total = batch * seq_len * heads * pairs;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    const int64_t pair = index % pairs;
    const int64_t head = (index / pairs) % heads;
    const int64_t seq = (index / (pairs * heads)) % seq_len;
    const int64_t batch_index = index / (pairs * heads * seq_len);
    const int64_t base = ((batch_index * seq_len + seq) * heads + head) * dim + pair * 2;
    const float c = static_cast<float>(cos[seq * pairs + pair]);
    const float s = static_cast<float>(sin[seq * pairs + pair]);
    const float grad_even = static_cast<float>(grad_out[base]);
    const float grad_odd = static_cast<float>(grad_out[base + 1]);
    grad_t[base] = static_cast<scalar_t>(grad_even * c + grad_odd * s);
    grad_t[base + 1] = static_cast<scalar_t>(-grad_even * s + grad_odd * c);
  }
}

at::Tensor rotary_forward_cuda(
    const at::Tensor& t,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  const c10::cuda::CUDAGuard device_guard(t.device());
  auto t_contiguous = t.contiguous();
  auto cos_contiguous = cos.contiguous();
  auto sin_contiguous = sin.contiguous();
  auto out = at::empty_like(t_contiguous);

  const auto batch = t_contiguous.size(0);
  const auto seq_len = t_contiguous.size(1);
  const auto heads = t_contiguous.size(2);
  const auto dim = t_contiguous.size(3);
  const int threads = 256;
  const int64_t total_pairs = batch * seq_len * heads * (dim / 2);
  const int blocks = static_cast<int>(std::min<int64_t>((total_pairs + threads - 1) / threads, 65535));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      t.scalar_type(),
      "rotary_forward_cuda",
      [&] {
        rotary_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            t_contiguous.data_ptr<scalar_t>(),
            cos_contiguous.data_ptr<scalar_t>(),
            sin_contiguous.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch,
            seq_len,
            heads,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}

at::Tensor rotary_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto cos_contiguous = cos.contiguous();
  auto sin_contiguous = sin.contiguous();
  auto grad_t = at::empty_like(grad_out_contiguous);

  const auto batch = grad_out_contiguous.size(0);
  const auto seq_len = grad_out_contiguous.size(1);
  const auto heads = grad_out_contiguous.size(2);
  const auto dim = grad_out_contiguous.size(3);
  const int threads = 256;
  const int64_t total_pairs = batch * seq_len * heads * (dim / 2);
  const int blocks = static_cast<int>(std::min<int64_t>((total_pairs + threads - 1) / threads, 65535));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "rotary_backward_cuda",
      [&] {
        rotary_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            cos_contiguous.data_ptr<scalar_t>(),
            sin_contiguous.data_ptr<scalar_t>(),
            grad_t.data_ptr<scalar_t>(),
            batch,
            seq_len,
            heads,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return grad_t;
}

template <typename scalar_t>
__global__ void rms_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gamma,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ normed,
    float* __restrict__ inv_norm,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  float sum_sq = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float value = static_cast<float>(x[row * dim + i]);
    sum_sq += value * value;
  }
  shared[tid] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float scale = sqrtf(static_cast<float>(dim));
  const float norm = fmaxf(sqrtf(shared[0]), 1.0e-12f);
  const float inv = 1.0f / norm;
  if (tid == 0) {
    inv_norm[row] = inv;
  }

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float normalized = static_cast<float>(x[row * dim + i]) * inv * scale;
    normed[row * dim + i] = static_cast<scalar_t>(normalized);
    out[row * dim + i] = static_cast<scalar_t>(normalized * static_cast<float>(gamma[i]));
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_input_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const scalar_t* __restrict__ gamma,
    scalar_t* __restrict__ grad_x,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(dim));
  const float inv = inv_norm[row];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = static_cast<float>(grad_out[row * dim + i]) * static_cast<float>(gamma[i]);
    dot_partial += grad_normed * static_cast<float>(normed[row * dim + i]);
  }
  shared[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = static_cast<float>(grad_out[row * dim + i]) * static_cast<float>(gamma[i]);
    const float normed_value = static_cast<float>(normed[row * dim + i]);
    const float value = inv * (grad_normed * scale - normed_value * (dot / scale));
    grad_x[row * dim + i] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_gamma_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    scalar_t* __restrict__ grad_gamma,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t i = blockIdx.x;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t row = tid; row < row_count; row += blockDim.x) {
    acc += static_cast<float>(grad_out[row * dim + i]) * static_cast<float>(normed[row * dim + i]);
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[i] = static_cast<scalar_t>(shared[0]);
  }
}

std::vector<at::Tensor> rms_norm_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contiguous = x.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = x_contiguous.size(-1);
  const auto row_count = x_contiguous.numel() / dim;
  auto out = at::empty_like(x_contiguous);
  auto normed = at::empty_like(x_contiguous);
  auto inv_norm = at::empty({row_count}, x_contiguous.options().dtype(at::kFloat));

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "rms_norm_forward_cuda",
      [&] {
        rms_norm_forward_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            gamma_contiguous.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, normed, inv_norm};
}

std::vector<at::Tensor> rms_norm_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto normed_contiguous = normed.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = grad_out_contiguous.size(-1);
  const auto row_count = grad_out_contiguous.numel() / dim;
  auto grad_x = at::empty_like(grad_out_contiguous);
  auto grad_gamma = at::empty_like(gamma_contiguous);

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "rms_norm_backward_cuda",
      [&] {
        rms_norm_backward_input_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            normed_contiguous.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma_contiguous.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            row_count,
            dim);
        rms_norm_backward_gamma_kernel<scalar_t><<<dim, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            normed_contiguous.data_ptr<scalar_t>(),
            grad_gamma.data_ptr<scalar_t>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma};
}

template <typename scalar_t>
__global__ void rms_norm_forward_fp32_gamma_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ gamma,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ normed,
    float* __restrict__ inv_norm,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  float sum_sq = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float value = static_cast<float>(x[row * dim + i]);
    sum_sq += value * value;
  }
  shared[tid] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float scale = sqrtf(static_cast<float>(dim));
  const float norm = fmaxf(sqrtf(shared[0]), 1.0e-12f);
  const float inv = 1.0f / norm;
  if (tid == 0) {
    inv_norm[row] = inv;
  }

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float normalized = static_cast<float>(x[row * dim + i]) * inv * scale;
    normed[row * dim + i] = static_cast<scalar_t>(normalized);
    out[row * dim + i] = static_cast<scalar_t>(normalized * gamma[i]);
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_fp32_gamma_input_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const float* __restrict__ gamma,
    scalar_t* __restrict__ grad_x,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(dim));
  const float inv = inv_norm[row];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = static_cast<float>(grad_out[row * dim + i]) * gamma[i];
    dot_partial += grad_normed * static_cast<float>(normed[row * dim + i]);
  }
  shared[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = static_cast<float>(grad_out[row * dim + i]) * gamma[i];
    const float normed_value = static_cast<float>(normed[row * dim + i]);
    const float value = inv * (grad_normed * scale - normed_value * (dot / scale));
    grad_x[row * dim + i] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_fp32_gamma_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    float* __restrict__ grad_gamma,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t i = blockIdx.x;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t row = tid; row < row_count; row += blockDim.x) {
    acc += static_cast<float>(grad_out[row * dim + i]) * static_cast<float>(normed[row * dim + i]);
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[i] = shared[0];
  }
}

std::vector<at::Tensor> rms_norm_forward_fp32_gamma_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contiguous = x.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = x_contiguous.size(-1);
  const auto row_count = x_contiguous.numel() / dim;
  auto out = at::empty_like(x_contiguous);
  auto normed = at::empty_like(x_contiguous);
  auto inv_norm = at::empty({row_count}, x_contiguous.options().dtype(at::kFloat));

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "rms_norm_forward_fp32_gamma_cuda",
      [&] {
        rms_norm_forward_fp32_gamma_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            gamma_contiguous.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, normed, inv_norm};
}

std::vector<at::Tensor> rms_norm_backward_fp32_gamma_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto normed_contiguous = normed.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = grad_out_contiguous.size(-1);
  const auto row_count = grad_out_contiguous.numel() / dim;
  auto grad_x = at::empty_like(grad_out_contiguous);
  auto grad_gamma = at::empty_like(gamma_contiguous);

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "rms_norm_backward_fp32_gamma_cuda",
      [&] {
        rms_norm_backward_fp32_gamma_input_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            normed_contiguous.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma_contiguous.data_ptr<float>(),
            grad_x.data_ptr<scalar_t>(),
            row_count,
            dim);
        rms_norm_backward_fp32_gamma_kernel<scalar_t><<<dim, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            normed_contiguous.data_ptr<scalar_t>(),
            grad_gamma.data_ptr<float>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma};
}

template <typename scalar_t>
__global__ void rms_norm_forward_mixed_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    float* __restrict__ normed,
    float* __restrict__ inv_norm,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  float sum_sq = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float value = static_cast<float>(x[row * dim + i]);
    sum_sq += value * value;
  }
  shared[tid] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float scale = sqrtf(static_cast<float>(dim));
  const float norm = fmaxf(sqrtf(shared[0]), 1.0e-12f);
  const float inv = 1.0f / norm;
  if (tid == 0) {
    inv_norm[row] = inv;
  }

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float normalized = static_cast<float>(x[row * dim + i]) * inv * scale;
    normed[row * dim + i] = normalized;
    out[row * dim + i] = normalized * gamma[i];
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_mixed_input_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const float* __restrict__ gamma,
    scalar_t* __restrict__ grad_x,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(dim));
  const float inv = inv_norm[row];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = grad_out[row * dim + i] * gamma[i];
    dot_partial += grad_normed * normed[row * dim + i];
  }
  shared[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = grad_out[row * dim + i] * gamma[i];
    const float value = inv * (grad_normed * scale - normed[row * dim + i] * (dot / scale));
    grad_x[row * dim + i] = static_cast<scalar_t>(value);
  }
}

__global__ void rms_norm_backward_mixed_gamma_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ normed,
    float* __restrict__ grad_gamma,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t i = blockIdx.x;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t row = tid; row < row_count; row += blockDim.x) {
    acc += grad_out[row * dim + i] * normed[row * dim + i];
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[i] = shared[0];
  }
}

std::vector<at::Tensor> rms_norm_forward_mixed_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contiguous = x.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = x_contiguous.size(-1);
  const auto row_count = x_contiguous.numel() / dim;
  auto float_options = x_contiguous.options().dtype(at::kFloat);
  auto out = at::empty(x_contiguous.sizes(), float_options);
  auto normed = at::empty(x_contiguous.sizes(), float_options);
  auto inv_norm = at::empty({row_count}, float_options);

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "rms_norm_forward_mixed_cuda",
      [&] {
        rms_norm_forward_mixed_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            gamma_contiguous.data_ptr<float>(),
            out.data_ptr<float>(),
            normed.data_ptr<float>(),
            inv_norm.data_ptr<float>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, normed, inv_norm};
}

std::vector<at::Tensor> rms_norm_backward_mixed_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& x_meta) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto normed_contiguous = normed.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = grad_out_contiguous.size(-1);
  const auto row_count = grad_out_contiguous.numel() / dim;
  auto grad_x = at::empty(grad_out_contiguous.sizes(), x_meta.options());
  auto grad_gamma = at::empty_like(gamma_contiguous);

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_meta.scalar_type(),
      "rms_norm_backward_mixed_cuda",
      [&] {
        rms_norm_backward_mixed_input_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<float>(),
            normed_contiguous.data_ptr<float>(),
            inv_norm.data_ptr<float>(),
            gamma_contiguous.data_ptr<float>(),
            grad_x.data_ptr<scalar_t>(),
            row_count,
            dim);
      });
  rms_norm_backward_mixed_gamma_kernel<<<dim, threads, shared_bytes, stream>>>(
      grad_out_contiguous.data_ptr<float>(),
      normed_contiguous.data_ptr<float>(),
      grad_gamma.data_ptr<float>(),
      row_count,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma};
}

template <typename scalar_t>
__global__ void attention_gate_compute_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ gate,
    int64_t row_count,
    int64_t heads,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    acc += static_cast<float>(x[row * dim + i]) * static_cast<float>(weight[head * dim + i]);
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float pre = shared[0] + static_cast<float>(bias[head]);
    gate[row * heads + head] = static_cast<scalar_t>(1.0f / (1.0f + expf(-pre)));
  }
}

template <typename scalar_t>
__global__ void attention_gate_apply_kernel(
    const scalar_t* __restrict__ attn,
    const scalar_t* __restrict__ gate,
    scalar_t* __restrict__ out,
    int64_t row_count,
    int64_t heads,
    int64_t dim_head) {
  const int64_t total = row_count * heads * dim_head;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    const int64_t dim = index % dim_head;
    const int64_t head = (index / dim_head) % heads;
    const int64_t row = index / (dim_head * heads);
    out[row * heads * dim_head + head * dim_head + dim] =
        static_cast<scalar_t>(static_cast<float>(attn[index]) * static_cast<float>(gate[row * heads + head]));
  }
}

template <typename scalar_t>
__global__ void attention_gate_backward_attn_gate_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ attn,
    const scalar_t* __restrict__ gate,
    scalar_t* __restrict__ grad_attn,
    scalar_t* __restrict__ grad_gate_pre,
    int64_t row_count,
    int64_t heads,
    int64_t dim_head) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int tid = threadIdx.x;
  const float gate_value = static_cast<float>(gate[row * heads + head]);

  float gate_grad_partial = 0.0f;
  for (int64_t d = tid; d < dim_head; d += blockDim.x) {
    const int64_t index = row * heads * dim_head + head * dim_head + d;
    const float grad = static_cast<float>(grad_out[index]);
    const float attn_value = static_cast<float>(attn[index]);
    grad_attn[index] = static_cast<scalar_t>(grad * gate_value);
    gate_grad_partial += grad * attn_value;
  }

  shared[tid] = gate_grad_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gate_pre[row * heads + head] = static_cast<scalar_t>(shared[0] * gate_value * (1.0f - gate_value));
  }
}

template <typename scalar_t>
__global__ void attention_gate_backward_x_kernel(
    const scalar_t* __restrict__ grad_gate_pre,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_x,
    int64_t row_count,
    int64_t heads,
    int64_t dim) {
  const int64_t row = blockIdx.x;
  const int64_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }

  float acc = 0.0f;
  for (int64_t head = 0; head < heads; ++head) {
    acc += static_cast<float>(grad_gate_pre[row * heads + head]) * static_cast<float>(weight[head * dim + i]);
  }
  grad_x[row * dim + i] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void attention_gate_backward_weight_bias_kernel(
    const scalar_t* __restrict__ grad_gate_pre,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    int64_t row_count,
    int64_t heads,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t head = blockIdx.x;
  const int64_t i = blockIdx.y;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t row = tid; row < row_count; row += blockDim.x) {
    const float grad = static_cast<float>(grad_gate_pre[row * heads + head]);
    acc += i == dim ? grad : grad * static_cast<float>(x[row * dim + i]);
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (i == dim) {
      grad_bias[head] = static_cast<scalar_t>(shared[0]);
    } else {
      grad_weight[head * dim + i] = static_cast<scalar_t>(shared[0]);
    }
  }
}

std::vector<at::Tensor> attention_gate_forward_cuda(
    const at::Tensor& attn,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  const c10::cuda::CUDAGuard device_guard(attn.device());
  auto attn_contiguous = attn.contiguous();
  auto x_contiguous = x.contiguous();
  auto weight_contiguous = weight.contiguous();
  auto bias_contiguous = bias.contiguous();

  const auto row_count = attn_contiguous.size(0);
  const auto heads = attn_contiguous.size(1);
  const auto dim_head = attn_contiguous.size(2);
  const auto dim = x_contiguous.size(1);

  auto out = at::empty({row_count, heads * dim_head}, attn_contiguous.options());
  auto gate = at::empty({row_count, heads}, attn_contiguous.options());
  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = row_count * heads * dim_head;
  const int apply_blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      attn.scalar_type(),
      "attention_gate_forward_cuda",
      [&] {
        attention_gate_compute_kernel<scalar_t><<<dim3(row_count, heads), threads, shared_bytes, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            weight_contiguous.data_ptr<scalar_t>(),
            bias_contiguous.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim);
        attention_gate_apply_kernel<scalar_t><<<apply_blocks, threads, 0, stream>>>(
            attn_contiguous.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim_head);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, gate, attn_contiguous, x_contiguous};
}

std::vector<at::Tensor> attention_gate_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& attn,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& gate) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto attn_contiguous = attn.contiguous();
  auto x_contiguous = x.contiguous();
  auto weight_contiguous = weight.contiguous();
  auto gate_contiguous = gate.contiguous();

  const auto row_count = attn_contiguous.size(0);
  const auto heads = attn_contiguous.size(1);
  const auto dim_head = attn_contiguous.size(2);
  const auto dim = x_contiguous.size(1);

  auto grad_attn = at::empty_like(attn_contiguous);
  auto grad_gate_pre = at::empty({row_count, heads}, grad_out_contiguous.options());
  auto grad_x = at::empty_like(x_contiguous);
  auto grad_weight = at::empty_like(weight_contiguous);
  auto grad_bias = at::empty({heads}, grad_out_contiguous.options());

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int dim_blocks = static_cast<int>((dim + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "attention_gate_backward_cuda",
      [&] {
        attention_gate_backward_attn_gate_kernel<scalar_t><<<dim3(row_count, heads), threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            attn_contiguous.data_ptr<scalar_t>(),
            gate_contiguous.data_ptr<scalar_t>(),
            grad_attn.data_ptr<scalar_t>(),
            grad_gate_pre.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim_head);
        attention_gate_backward_x_kernel<scalar_t><<<dim3(row_count, dim_blocks), threads, 0, stream>>>(
            grad_gate_pre.data_ptr<scalar_t>(),
            weight_contiguous.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim);
        attention_gate_backward_weight_bias_kernel<scalar_t><<<dim3(heads, dim + 1), threads, shared_bytes, stream>>>(
            grad_gate_pre.data_ptr<scalar_t>(),
            x_contiguous.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_attn, grad_x, grad_weight, grad_bias};
}

cudaDataType_t cuda_data_type_for(at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf:
      return CUDA_R_16F;
    case at::kBFloat16:
      return CUDA_R_16BF;
    case at::kFloat:
      return CUDA_R_32F;
    default:
      TORCH_CHECK(false, "unsupported dtype for grouped GEMM");
  }
}

cublasComputeType_t cublas_compute_type_for(at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf:
      return CUBLAS_COMPUTE_32F_FAST_16F;
    case at::kBFloat16:
      return CUBLAS_COMPUTE_32F_FAST_16BF;
    case at::kFloat:
      return CUBLAS_COMPUTE_32F;
    default:
      TORCH_CHECK(false, "unsupported dtype for grouped GEMM");
  }
}

void check_cublas_status(cublasStatus_t status, const char* name) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, name, " failed with cublas status ", static_cast<int>(status));
}

std::vector<int64_t> dim_offsets_from_inputs(const std::vector<int64_t>& dim_inputs) {
  std::vector<int64_t> offsets;
  offsets.reserve(dim_inputs.size() + 1);
  offsets.push_back(0);
  for (const auto dim : dim_inputs) {
    TORCH_CHECK(dim > 0, "dim_inputs must be positive");
    offsets.push_back(offsets.back() + dim);
  }
  return offsets;
}

struct GroupedGemmPlan {
  std::vector<int> m_array;
  std::vector<int> n_array;
  std::vector<int> k_array;
  std::vector<int> lda_array;
  std::vector<int> ldb_array;
  std::vector<int> ldc_array;
  std::vector<int> group_size;
  std::vector<const void*> a_array;
  std::vector<const void*> b_array;
  std::vector<void*> c_array;
  std::vector<cublasOperation_t> transa_array;
  std::vector<cublasOperation_t> transb_array;
};

int find_or_add_group(std::vector<int>& group_keys, int key) {
  for (size_t index = 0; index < group_keys.size(); ++index) {
    if (group_keys[index] == key) {
      return static_cast<int>(index);
    }
  }
  group_keys.push_back(key);
  return static_cast<int>(group_keys.size() - 1);
}

template <typename scalar_t>
GroupedGemmPlan make_mask_forward_plan(
    const scalar_t* x,
    const std::vector<at::Tensor>& weights,
    scalar_t* pre,
    const std::vector<int64_t>& dim_inputs,
    const std::vector<int64_t>& offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t hidden,
    int64_t total_pre) {
  std::vector<int> group_keys;
  std::vector<std::vector<int64_t>> grouped_bands;
  for (int64_t band = 0; band < band_count; ++band) {
    const int pre_out = static_cast<int>(dim_inputs[band] * 2);
    const int group = find_or_add_group(group_keys, pre_out);
    if (group == static_cast<int>(grouped_bands.size())) {
      grouped_bands.emplace_back();
    }
    grouped_bands[group].push_back(band);
  }

  GroupedGemmPlan plan;
  const int group_count = static_cast<int>(group_keys.size());
  plan.m_array.reserve(group_count);
  plan.n_array.reserve(group_count);
  plan.k_array.reserve(group_count);
  plan.lda_array.reserve(group_count);
  plan.ldb_array.reserve(group_count);
  plan.ldc_array.reserve(group_count);
  plan.group_size.reserve(group_count);
  plan.transa_array.reserve(group_count);
  plan.transb_array.reserve(group_count);

  for (int group = 0; group < group_count; ++group) {
    const int pre_out = group_keys[group];
    plan.transa_array.push_back(CUBLAS_OP_T);
    plan.transb_array.push_back(CUBLAS_OP_N);
    plan.m_array.push_back(pre_out);
    plan.n_array.push_back(static_cast<int>(m_count));
    plan.k_array.push_back(static_cast<int>(hidden));
    plan.lda_array.push_back(static_cast<int>(hidden));
    plan.ldb_array.push_back(static_cast<int>(band_count * hidden));
    plan.ldc_array.push_back(static_cast<int>(total_pre));
    plan.group_size.push_back(static_cast<int>(grouped_bands[group].size()));
    for (const auto band : grouped_bands[group]) {
      plan.a_array.push_back(weights[band].data_ptr<scalar_t>());
      plan.b_array.push_back(x + band * hidden);
      plan.c_array.push_back(pre + offsets[band] * 2);
    }
  }
  return plan;
}

template <typename scalar_t>
GroupedGemmPlan make_mask_grad_x_plan(
    const scalar_t* grad_pre,
    const std::vector<at::Tensor>& weights,
    scalar_t* grad_x,
    const std::vector<int64_t>& dim_inputs,
    const std::vector<int64_t>& offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t hidden,
    int64_t total_pre) {
  std::vector<int> group_keys;
  std::vector<std::vector<int64_t>> grouped_bands;
  for (int64_t band = 0; band < band_count; ++band) {
    const int pre_out = static_cast<int>(dim_inputs[band] * 2);
    const int group = find_or_add_group(group_keys, pre_out);
    if (group == static_cast<int>(grouped_bands.size())) {
      grouped_bands.emplace_back();
    }
    grouped_bands[group].push_back(band);
  }

  GroupedGemmPlan plan;
  const int group_count = static_cast<int>(group_keys.size());
  for (int group = 0; group < group_count; ++group) {
    const int pre_out = group_keys[group];
    plan.transa_array.push_back(CUBLAS_OP_N);
    plan.transb_array.push_back(CUBLAS_OP_N);
    plan.m_array.push_back(static_cast<int>(hidden));
    plan.n_array.push_back(static_cast<int>(m_count));
    plan.k_array.push_back(pre_out);
    plan.lda_array.push_back(static_cast<int>(hidden));
    plan.ldb_array.push_back(static_cast<int>(total_pre));
    plan.ldc_array.push_back(static_cast<int>(band_count * hidden));
    plan.group_size.push_back(static_cast<int>(grouped_bands[group].size()));
    for (const auto band : grouped_bands[group]) {
      plan.a_array.push_back(weights[band].data_ptr<scalar_t>());
      plan.b_array.push_back(grad_pre + offsets[band] * 2);
      plan.c_array.push_back(grad_x + band * hidden);
    }
  }
  return plan;
}

template <typename scalar_t>
GroupedGemmPlan make_mask_grad_weight_plan(
    const scalar_t* x,
    const scalar_t* grad_pre,
    const std::vector<at::Tensor>& grad_weights,
    const std::vector<int64_t>& dim_inputs,
    const std::vector<int64_t>& offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t hidden,
    int64_t total_pre) {
  std::vector<int> group_keys;
  std::vector<std::vector<int64_t>> grouped_bands;
  for (int64_t band = 0; band < band_count; ++band) {
    const int pre_out = static_cast<int>(dim_inputs[band] * 2);
    const int group = find_or_add_group(group_keys, pre_out);
    if (group == static_cast<int>(grouped_bands.size())) {
      grouped_bands.emplace_back();
    }
    grouped_bands[group].push_back(band);
  }

  GroupedGemmPlan plan;
  const int group_count = static_cast<int>(group_keys.size());
  for (int group = 0; group < group_count; ++group) {
    const int pre_out = group_keys[group];
    plan.transa_array.push_back(CUBLAS_OP_N);
    plan.transb_array.push_back(CUBLAS_OP_T);
    plan.m_array.push_back(static_cast<int>(hidden));
    plan.n_array.push_back(pre_out);
    plan.k_array.push_back(static_cast<int>(m_count));
    plan.lda_array.push_back(static_cast<int>(band_count * hidden));
    plan.ldb_array.push_back(static_cast<int>(total_pre));
    plan.ldc_array.push_back(static_cast<int>(hidden));
    plan.group_size.push_back(static_cast<int>(grouped_bands[group].size()));
    for (const auto band : grouped_bands[group]) {
      plan.a_array.push_back(x + band * hidden);
      plan.b_array.push_back(grad_pre + offsets[band] * 2);
      plan.c_array.push_back(grad_weights[band].data_ptr<scalar_t>());
    }
  }
  return plan;
}

void run_grouped_gemm(
    const GroupedGemmPlan& plan,
    cudaDataType_t data_type,
    cublasComputeType_t compute_type,
    const char* name) {
  if (plan.group_size.empty()) {
    return;
  }
  std::vector<float> alpha(plan.group_size.size(), 1.0f);
  std::vector<float> beta(plan.group_size.size(), 0.0f);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  auto stream = at::cuda::getCurrentCUDAStream();
  check_cublas_status(cublasSetStream(handle, stream), "cublasSetStream");
  const auto status = cublasGemmGroupedBatchedEx(
      handle,
      plan.transa_array.data(),
      plan.transb_array.data(),
      plan.m_array.data(),
      plan.n_array.data(),
      plan.k_array.data(),
      alpha.data(),
      plan.a_array.data(),
      data_type,
      plan.lda_array.data(),
      plan.b_array.data(),
      data_type,
      plan.ldb_array.data(),
      beta.data(),
      plan.c_array.data(),
      data_type,
      plan.ldc_array.data(),
      static_cast<int>(plan.group_size.size()),
      plan.group_size.data(),
      compute_type);
  check_cublas_status(status, name);
}

template <typename scalar_t>
__global__ void mask_estimator_final_glu_forward_kernel(
    scalar_t* __restrict__ pre,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t total,
    int64_t total_pre) {
  const int64_t m = blockIdx.x;
  const int64_t band = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t dim = end - start;
  const int64_t pre_start = start * 2;
  for (int64_t local = tid; local < dim; local += blockDim.x) {
    const int64_t pre_left = m * total_pre + pre_start + local;
    const int64_t pre_gate = m * total_pre + pre_start + dim + local;
    const float left = static_cast<float>(pre[pre_left]) + static_cast<float>(bias[pre_start + local]);
    const float gate_pre = static_cast<float>(pre[pre_gate]) + static_cast<float>(bias[pre_start + dim + local]);
    const float gate = 1.0f / (1.0f + expf(-gate_pre));
    pre[pre_left] = static_cast<scalar_t>(left);
    pre[pre_gate] = static_cast<scalar_t>(gate_pre);
    out[m * total + start + local] = static_cast<scalar_t>(left * gate);
  }
}

template <typename scalar_t>
__global__ void mask_estimator_final_glu_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ pre,
    scalar_t* __restrict__ grad_pre,
    scalar_t* __restrict__ grad_bias,
    const int64_t* __restrict__ offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t total,
    int64_t total_pre) {
  const int64_t band = blockIdx.x;
  const int64_t local = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t dim = end - start;
  if (local >= dim) {
    return;
  }
  const int64_t pre_left_offset = start * 2 + local;
  const int64_t pre_gate_offset = start * 2 + dim + local;

  float bias_left = 0.0f;
  float bias_gate = 0.0f;
  for (int64_t m = tid; m < m_count; m += blockDim.x) {
    const float left = static_cast<float>(pre[m * total_pre + pre_left_offset]);
    const float gate_pre = static_cast<float>(pre[m * total_pre + pre_gate_offset]);
    const float gate = 1.0f / (1.0f + expf(-gate_pre));
    const float grad = static_cast<float>(grad_out[m * total + start + local]);
    const float grad_left = grad * gate;
    const float grad_gate = grad * left * gate * (1.0f - gate);
    grad_pre[m * total_pre + pre_left_offset] = static_cast<scalar_t>(grad_left);
    grad_pre[m * total_pre + pre_gate_offset] = static_cast<scalar_t>(grad_gate);
    bias_left += grad_left;
    bias_gate += grad_gate;
  }

  extern __shared__ float shared[];
  float* shared_left = shared;
  float* shared_gate = shared + blockDim.x;
  shared_left[tid] = bias_left;
  shared_gate[tid] = bias_gate;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_left[tid] += shared_left[tid + stride];
      shared_gate[tid] += shared_gate[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_bias[pre_left_offset] = static_cast<scalar_t>(shared_left[0]);
    grad_bias[pre_gate_offset] = static_cast<scalar_t>(shared_gate[0]);
  }
}

std::vector<at::Tensor> mask_estimator_final_forward_cuda(
    const at::Tensor& x,
    const std::vector<at::Tensor>& weights,
    const at::Tensor& bias,
    const std::vector<int64_t>& dim_inputs,
    const at::Tensor& dim_offsets) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contiguous = x.contiguous();
  auto bias_contiguous = bias.contiguous();

  const auto m_count = x_contiguous.size(0);
  const auto band_count = x_contiguous.size(1);
  const auto hidden = x_contiguous.size(2);
  const auto offsets = dim_offsets_from_inputs(dim_inputs);
  const auto total = offsets.back();
  const auto total_pre = total * 2;

  auto pre = at::empty({m_count, total_pre}, x_contiguous.options());
  auto out = at::empty({m_count, total}, x_contiguous.options());

  const auto data_type = cuda_data_type_for(x.scalar_type());
  const auto compute_type = cublas_compute_type_for(x.scalar_type());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "mask_estimator_final_forward_cuda",
      [&] {
        auto plan = make_mask_forward_plan<scalar_t>(
            x_contiguous.data_ptr<scalar_t>(),
            weights,
            pre.data_ptr<scalar_t>(),
            dim_inputs,
            offsets,
            m_count,
            band_count,
            hidden,
            total_pre);
        run_grouped_gemm(plan, data_type, compute_type, "mask_estimator_final_forward_grouped_gemm");

        const int threads = 256;
        const auto stream = at::cuda::getCurrentCUDAStream();
        mask_estimator_final_glu_forward_kernel<scalar_t><<<dim3(m_count, band_count), threads, 0, stream>>>(
            pre.data_ptr<scalar_t>(),
            bias_contiguous.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            dim_offsets.data_ptr<int64_t>(),
            m_count,
            band_count,
            total,
            total_pre);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, pre, x_contiguous};
}

std::vector<at::Tensor> mask_estimator_final_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& pre,
    const std::vector<at::Tensor>& weights,
    const std::vector<int64_t>& dim_inputs,
    const at::Tensor& dim_offsets) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto x_contiguous = x.contiguous();
  auto pre_contiguous = pre.contiguous();

  const auto m_count = x_contiguous.size(0);
  const auto band_count = x_contiguous.size(1);
  const auto hidden = x_contiguous.size(2);
  const auto offsets = dim_offsets_from_inputs(dim_inputs);
  const auto total = offsets.back();
  const auto total_pre = total * 2;

  auto grad_pre = at::empty({m_count, total_pre}, grad_out_contiguous.options());
  auto grad_bias = at::empty({total_pre}, grad_out_contiguous.options());
  auto grad_x = at::empty_like(x_contiguous);
  std::vector<at::Tensor> grad_weights;
  grad_weights.reserve(weights.size());
  for (const auto& weight : weights) {
    grad_weights.push_back(at::empty_like(weight));
  }

  const auto data_type = cuda_data_type_for(grad_out.scalar_type());
  const auto compute_type = cublas_compute_type_for(grad_out.scalar_type());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "mask_estimator_final_backward_cuda",
      [&] {
        const int threads = 256;
        const size_t shared_bytes = threads * sizeof(float) * 2;
        const auto stream = at::cuda::getCurrentCUDAStream();
        const int64_t max_dim = *std::max_element(dim_inputs.begin(), dim_inputs.end());
        mask_estimator_final_glu_backward_kernel<scalar_t><<<dim3(band_count, max_dim), threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            pre_contiguous.data_ptr<scalar_t>(),
            grad_pre.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            dim_offsets.data_ptr<int64_t>(),
            m_count,
            band_count,
            total,
            total_pre);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto grad_x_plan = make_mask_grad_x_plan<scalar_t>(
            grad_pre.data_ptr<scalar_t>(),
            weights,
            grad_x.data_ptr<scalar_t>(),
            dim_inputs,
            offsets,
            m_count,
            band_count,
            hidden,
            total_pre);
        run_grouped_gemm(grad_x_plan, data_type, compute_type, "mask_estimator_final_grad_x_grouped_gemm");

        auto grad_weight_plan = make_mask_grad_weight_plan<scalar_t>(
            x_contiguous.data_ptr<scalar_t>(),
            grad_pre.data_ptr<scalar_t>(),
            grad_weights,
            dim_inputs,
            offsets,
            m_count,
            band_count,
            hidden,
            total_pre);
        run_grouped_gemm(grad_weight_plan, data_type, compute_type, "mask_estimator_final_grad_weight_grouped_gemm");
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<at::Tensor> result;
  result.reserve(2 + grad_weights.size());
  result.push_back(grad_x);
  result.push_back(grad_bias);
  for (auto& grad_weight : grad_weights) {
    result.push_back(grad_weight);
  }
  return result;
}

template <typename scalar_t>
__global__ void mask_estimator_final_direct_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* const* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ pre,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t hidden,
    int64_t total,
    int64_t total_pre) {
  const int64_t m = blockIdx.x;
  const int64_t band = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t dim = end - start;
  const int64_t pre_start = start * 2;
  const scalar_t* weight = weights[band];

  for (int64_t local = tid; local < dim; local += blockDim.x) {
    float left = static_cast<float>(bias[pre_start + local]);
    float gate_pre = static_cast<float>(bias[pre_start + dim + local]);
    for (int64_t h = 0; h < hidden; ++h) {
      const float x_value = static_cast<float>(x[(m * band_count + band) * hidden + h]);
      left += x_value * static_cast<float>(weight[local * hidden + h]);
      gate_pre += x_value * static_cast<float>(weight[(dim + local) * hidden + h]);
    }
    const int64_t pre_left = m * total_pre + pre_start + local;
    const int64_t pre_gate = m * total_pre + pre_start + dim + local;
    const float gate = 1.0f / (1.0f + expf(-gate_pre));
    pre[pre_left] = static_cast<scalar_t>(left);
    pre[pre_gate] = static_cast<scalar_t>(gate_pre);
    out[m * total + start + local] = static_cast<scalar_t>(left * gate);
  }
}

template <typename scalar_t>
__global__ void mask_estimator_final_direct_grad_x_kernel(
    const scalar_t* __restrict__ grad_pre,
    const scalar_t* const* __restrict__ weights,
    scalar_t* __restrict__ grad_x,
    const int64_t* __restrict__ offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t hidden,
    int64_t total_pre) {
  const int64_t m = blockIdx.x;
  const int64_t band = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t dim = end - start;
  const int64_t pre_start = start * 2;
  const scalar_t* weight = weights[band];

  for (int64_t h = tid; h < hidden; h += blockDim.x) {
    float acc = 0.0f;
    for (int64_t local = 0; local < dim; ++local) {
      acc += static_cast<float>(grad_pre[m * total_pre + pre_start + local]) *
             static_cast<float>(weight[local * hidden + h]);
      acc += static_cast<float>(grad_pre[m * total_pre + pre_start + dim + local]) *
             static_cast<float>(weight[(dim + local) * hidden + h]);
    }
    grad_x[(m * band_count + band) * hidden + h] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void mask_estimator_final_direct_glu_grad_x_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ pre,
    const scalar_t* const* __restrict__ weights,
    scalar_t* __restrict__ grad_pre,
    scalar_t* __restrict__ grad_bias,
    scalar_t* __restrict__ grad_x,
    const int64_t* __restrict__ offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t hidden,
    int64_t total,
    int64_t total_pre) {
  extern __shared__ float shared[];
  const int64_t m = blockIdx.x;
  const int64_t band = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t dim = end - start;
  const int64_t pre_start = start * 2;
  const scalar_t* weight = weights[band];

  if (blockIdx.z == 0) {
    for (int64_t local = tid; local < dim; local += blockDim.x) {
      const int64_t pre_left_offset = pre_start + local;
      const int64_t pre_gate_offset = pre_start + dim + local;
      const float left = static_cast<float>(pre[m * total_pre + pre_left_offset]);
      const float gate_pre = static_cast<float>(pre[m * total_pre + pre_gate_offset]);
      const float gate = 1.0f / (1.0f + expf(-gate_pre));
      const float grad = static_cast<float>(grad_out[m * total + start + local]);
      grad_pre[m * total_pre + pre_left_offset] = static_cast<scalar_t>(grad * gate);
      grad_pre[m * total_pre + pre_gate_offset] = static_cast<scalar_t>(grad * left * gate * (1.0f - gate));
    }
    for (int64_t h = tid; h < hidden; h += blockDim.x) {
      float acc = 0.0f;
      for (int64_t local = 0; local < dim; ++local) {
        const int64_t pre_left_offset = pre_start + local;
        const int64_t pre_gate_offset = pre_start + dim + local;
        const float left = static_cast<float>(pre[m * total_pre + pre_left_offset]);
        const float gate_pre = static_cast<float>(pre[m * total_pre + pre_gate_offset]);
        const float gate = 1.0f / (1.0f + expf(-gate_pre));
        const float grad = static_cast<float>(grad_out[m * total + start + local]);
        const float grad_left = grad * gate;
        const float grad_gate = grad * left * gate * (1.0f - gate);
        acc += grad_left * static_cast<float>(weight[local * hidden + h]);
        acc += grad_gate * static_cast<float>(weight[(dim + local) * hidden + h]);
      }
      grad_x[(m * band_count + band) * hidden + h] = static_cast<scalar_t>(acc);
    }
  }

  if (m == 0) {
    for (int64_t local = blockIdx.z; local < dim; local += gridDim.z) {
      const int64_t pre_left_offset = pre_start + local;
      const int64_t pre_gate_offset = pre_start + dim + local;
      float bias_left = 0.0f;
      float bias_gate = 0.0f;
      for (int64_t row = tid; row < m_count; row += blockDim.x) {
        const float left = static_cast<float>(pre[row * total_pre + pre_left_offset]);
        const float gate_pre = static_cast<float>(pre[row * total_pre + pre_gate_offset]);
        const float gate = 1.0f / (1.0f + expf(-gate_pre));
        const float grad = static_cast<float>(grad_out[row * total + start + local]);
        bias_left += grad * gate;
        bias_gate += grad * left * gate * (1.0f - gate);
      }
      float* shared_left = shared;
      float* shared_gate = shared + blockDim.x;
      shared_left[tid] = bias_left;
      shared_gate[tid] = bias_gate;
      __syncthreads();
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          shared_left[tid] += shared_left[tid + stride];
          shared_gate[tid] += shared_gate[tid + stride];
        }
        __syncthreads();
      }
      if (tid == 0) {
        grad_bias[pre_left_offset] = static_cast<scalar_t>(shared_left[0]);
        grad_bias[pre_gate_offset] = static_cast<scalar_t>(shared_gate[0]);
      }
    }
  }
}

template <typename scalar_t>
__global__ void mask_estimator_final_direct_grad_weight_kernel(
    const scalar_t* __restrict__ grad_pre,
    const scalar_t* __restrict__ x,
    scalar_t* const* __restrict__ grad_weights,
    const int64_t* __restrict__ offsets,
    int64_t m_count,
    int64_t band_count,
    int64_t hidden,
    int64_t max_pre_out,
    int64_t total_pre) {
  const int64_t band = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t start = offsets[band];
  const int64_t end = offsets[band + 1];
  const int64_t dim = end - start;
  const int64_t pre_out = dim * 2;
  if (row >= pre_out || row >= max_pre_out) {
    return;
  }
  const int64_t pre_start = start * 2;
  scalar_t* grad_weight = grad_weights[band];

  for (int64_t h = tid; h < hidden; h += blockDim.x) {
    float acc = 0.0f;
    for (int64_t m = 0; m < m_count; ++m) {
      acc += static_cast<float>(grad_pre[m * total_pre + pre_start + row]) *
             static_cast<float>(x[(m * band_count + band) * hidden + h]);
    }
    grad_weight[row * hidden + h] = static_cast<scalar_t>(acc);
  }
}

std::vector<at::Tensor> mask_estimator_final_direct_forward_cuda(
    const at::Tensor& x,
    const std::vector<at::Tensor>& weights,
    const at::Tensor& bias,
    const std::vector<int64_t>& dim_inputs,
    const at::Tensor& dim_offsets) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contiguous = x.contiguous();
  auto bias_contiguous = bias.contiguous();

  const auto m_count = x_contiguous.size(0);
  const auto band_count = x_contiguous.size(1);
  const auto hidden = x_contiguous.size(2);
  const auto offsets = dim_offsets_from_inputs(dim_inputs);
  const auto total = offsets.back();
  const auto total_pre = total * 2;

  auto pre = at::empty({m_count, total_pre}, x_contiguous.options());
  auto out = at::empty({m_count, total}, x_contiguous.options());

  const int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "mask_estimator_final_direct_forward_cuda",
      [&] {
        std::vector<at::Tensor> weight_contiguous;
        std::vector<const scalar_t*> weight_ptrs;
        weight_contiguous.reserve(weights.size());
        weight_ptrs.reserve(weights.size());
        for (size_t index = 0; index < weights.size(); ++index) {
          weight_contiguous.push_back(contiguous_if_needed(weights[index]));
          weight_ptrs.push_back(weight_contiguous.back().data_ptr<scalar_t>());
        }
        auto weight_ptr_tensor = tensor_from_host_ptrs(weight_ptrs, x_contiguous.options());
        mask_estimator_final_direct_forward_kernel<scalar_t><<<dim3(m_count, band_count), threads, 0, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t* const*>(weight_ptr_tensor.data_ptr<int64_t>()),
            bias_contiguous.data_ptr<scalar_t>(),
            pre.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            dim_offsets.data_ptr<int64_t>(),
            m_count,
            band_count,
            hidden,
            total,
            total_pre);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, pre, x_contiguous};
}

std::vector<at::Tensor> mask_estimator_final_direct_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& pre,
    const std::vector<at::Tensor>& weights,
    const std::vector<int64_t>& dim_inputs,
    const at::Tensor& dim_offsets) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto x_contiguous = x.contiguous();
  auto pre_contiguous = pre.contiguous();

  const auto m_count = x_contiguous.size(0);
  const auto band_count = x_contiguous.size(1);
  const auto hidden = x_contiguous.size(2);
  const auto offsets = dim_offsets_from_inputs(dim_inputs);
  const auto total = offsets.back();
  const auto total_pre = total * 2;
  const auto max_pre_out = *std::max_element(dim_inputs.begin(), dim_inputs.end()) * 2;

  auto grad_pre = at::empty({m_count, total_pre}, grad_out_contiguous.options());
  auto grad_bias = at::empty({total_pre}, grad_out_contiguous.options());
  auto grad_x = at::empty_like(x_contiguous);
  std::vector<at::Tensor> grad_weights;
  grad_weights.reserve(weights.size());
  for (const auto& weight : weights) {
    grad_weights.push_back(at::empty_like(weight));
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "mask_estimator_final_direct_backward_cuda",
      [&] {
        const int threads = 256;
        const size_t shared_bytes = threads * sizeof(float) * 2;
        const auto stream = at::cuda::getCurrentCUDAStream();

        std::vector<at::Tensor> weight_contiguous;
        std::vector<const scalar_t*> weight_ptrs;
        std::vector<const scalar_t*> grad_weight_ptrs;
        weight_contiguous.reserve(weights.size());
        weight_ptrs.reserve(weights.size());
        grad_weight_ptrs.reserve(weights.size());
        for (size_t index = 0; index < weights.size(); ++index) {
          weight_contiguous.push_back(contiguous_if_needed(weights[index]));
          weight_ptrs.push_back(weight_contiguous.back().data_ptr<scalar_t>());
          grad_weight_ptrs.push_back(grad_weights[index].data_ptr<scalar_t>());
        }
        auto weight_ptr_tensor = tensor_from_host_ptrs(weight_ptrs, grad_out_contiguous.options());
        auto grad_weight_ptr_tensor = tensor_from_host_ptrs(grad_weight_ptrs, grad_out_contiguous.options());

        mask_estimator_final_direct_glu_grad_x_kernel<scalar_t>
            <<<dim3(m_count, band_count, max_pre_out / 2), threads, shared_bytes, stream>>>(
                grad_out_contiguous.data_ptr<scalar_t>(),
                pre_contiguous.data_ptr<scalar_t>(),
                reinterpret_cast<const scalar_t* const*>(weight_ptr_tensor.data_ptr<int64_t>()),
                grad_pre.data_ptr<scalar_t>(),
                grad_bias.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                dim_offsets.data_ptr<int64_t>(),
                m_count,
                band_count,
                hidden,
                total,
                total_pre);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        mask_estimator_final_direct_grad_weight_kernel<scalar_t><<<dim3(band_count, max_pre_out), threads, 0, stream>>>(
            grad_pre.data_ptr<scalar_t>(),
            x_contiguous.data_ptr<scalar_t>(),
            reinterpret_cast<scalar_t* const*>(grad_weight_ptr_tensor.data_ptr<int64_t>()),
            dim_offsets.data_ptr<int64_t>(),
            m_count,
            band_count,
            hidden,
            max_pre_out,
            total_pre);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<at::Tensor> result;
  result.reserve(2 + grad_weights.size());
  result.push_back(grad_x);
  result.push_back(grad_bias);
  for (auto& grad_weight : grad_weights) {
    result.push_back(grad_weight);
  }
  return result;
}
