#include "inference/kernels.h"

#include <type_traits>
#include <cmath>
#include <climits>
#include <cstdint>
#include <mma.h>

using namespace nvcuda;

template <typename T>
__device__ float to_float(T value);

template <>
__device__ float to_float<float>(float value) { return value; }

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ T from_float(float value);

template <>
__device__ float from_float<float>(float value) { return value; }

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void residual_add_kernel(const T* a, const T* b, T* out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

template <typename T>
__global__ void embedding_kernel(const T* table, const int* ids, T* out,
                                 int rows, int hidden) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const T* src = table + static_cast<size_t>(ids[row]) * hidden;
  T* dst = out + static_cast<size_t>(row) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

template <typename T>
__global__ void rmsnorm_kernel(const T* x, const T* weight, T* out,
                               int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float partial[256];
  const T* src = x + static_cast<size_t>(row) * hidden;
  T* dst = out + static_cast<size_t>(row) * hidden;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = to_float(src[i]);
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(partial[0] / hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float normalized = to_float(src[i]) * inv;
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      normalized = to_float(from_float<T>(normalized));
    }
    dst[i] = from_float<T>(normalized * to_float(weight[i]));
  }
}

template <typename T>
__global__ void residual_rmsnorm_kernel(const T* x, const T* proj,
                                        const T* weight, T* residual,
                                        T* normalized, int rows, int hidden,
                                        float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float partial[256];
  const size_t base = static_cast<size_t>(row) * hidden;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const T value = from_float<T>(to_float(x[base + i]) + to_float(proj[base + i]));
    residual[base + i] = value;
    const float f = to_float(value);
    sum += f * f;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(partial[0] / hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float value = to_float(residual[base + i]) * inv;
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      value = to_float(from_float<T>(value));
    }
    normalized[base + i] = from_float<T>(value * to_float(weight[i]));
  }
}

template <typename T>
__global__ void project_rope_cache_kernel(
    const T* x, const T* weight, T* cache,
    const float* cos_table, const float* sin_table,
    int start_pos, int rows, int n_kv_heads, int head_dim, int hidden) {
  const int pairs_per_row = n_kv_heads * (head_dim / 2);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair_index = blockIdx.x * (blockDim.x >> 5) + warp;
  if (pair_index >= rows * pairs_per_row) return;
  const int row = pair_index / pairs_per_row;
  const int pair = pair_index % pairs_per_row;
  const int head = pair / (head_dim / 2);
  const int i = pair % (head_dim / 2);
  const int j = i + head_dim / 2;
  const int n0 = head * head_dim + i;
  const int n1 = head * head_dim + j;
  const T* input = x + static_cast<size_t>(row) * hidden;
  float a = 0.0f;
  float b = 0.0f;
  const T* w0 = weight + static_cast<size_t>(n0) * hidden;
  const T* w1 = weight + static_cast<size_t>(n1) * hidden;
  const bool aligned = hidden % 4 == 0 &&
      (reinterpret_cast<uintptr_t>(input) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(w0) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(w1) & 15) == 0;
  if (aligned) {
    for (int group = lane; group < hidden / 4; group += 32) {
      if constexpr (std::is_same<T, float>::value) {
        const float4 xv = reinterpret_cast<const float4*>(input)[group];
        const float4 a0 = reinterpret_cast<const float4*>(w0)[group];
        const float4 b0 = reinterpret_cast<const float4*>(w1)[group];
        a += xv.x*a0.x + xv.y*a0.y + xv.z*a0.z + xv.w*a0.w;
        b += xv.x*b0.x + xv.y*b0.y + xv.z*b0.z + xv.w*b0.w;
      } else {
        const uint2 xp = reinterpret_cast<const uint2*>(input)[group];
        const uint2 a_pack = reinterpret_cast<const uint2*>(w0)[group];
        const uint2 b_pack = reinterpret_cast<const uint2*>(w1)[group];
        const auto* xv = reinterpret_cast<const __nv_bfloat16*>(&xp);
        const auto* av = reinterpret_cast<const __nv_bfloat16*>(&a_pack);
        const auto* bv = reinterpret_cast<const __nv_bfloat16*>(&b_pack);
        for (int j = 0; j < 4; ++j) {
          a += to_float(xv[j]) * to_float(av[j]);
          b += to_float(xv[j]) * to_float(bv[j]);
        }
      }
    }
  } else {
    for (int k = lane; k < hidden; k += 32) {
      const float xv = to_float(input[k]);
      a += xv * to_float(w0[k]);
      b += xv * to_float(w1[k]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    a += __shfl_down_sync(0xffffffff, a, offset);
    b += __shfl_down_sync(0xffffffff, b, offset);
  }
  if (lane != 0) return;
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    a = to_float(from_float<T>(a));
    b = to_float(from_float<T>(b));
  }
  const size_t table_idx = static_cast<size_t>(start_pos + row) * (head_dim / 2) + i;
  const float c = cos_table[table_idx];
  const float s = sin_table[table_idx];
  T* dst = cache + static_cast<size_t>(row) * pairs_per_row * 2 + head * head_dim;
  dst[i] = from_float<T>(a * c - b * s);
  dst[j] = from_float<T>(b * c + a * s);
}

template <typename T>
__global__ void linear_swiglu_kernel(
    const T* x, const T* gate_weight, const T* up_weight, T* out, int rows,
    int intermediate, int hidden) {
  const size_t count = static_cast<size_t>(rows) * intermediate;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const size_t index = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + warp;
  if (index >= count) return;
  const int row = static_cast<int>(index / intermediate);
  const int col = static_cast<int>(index % intermediate);
  const T* input = x + static_cast<size_t>(row) * hidden;
  const T* gw = gate_weight + static_cast<size_t>(col) * hidden;
  const T* uw = up_weight + static_cast<size_t>(col) * hidden;
  float gate = 0.0f;
  float up = 0.0f;
  if constexpr (std::is_same<T, float>::value) {
    const bool aligned = hidden % 4 == 0 &&
        (reinterpret_cast<uintptr_t>(input) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(gw) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(uw) & 15) == 0;
    if (aligned) {
      for (int group = lane; group < hidden / 4; group += 32) {
        const float4 xv = reinterpret_cast<const float4*>(input)[group];
        const float4 gv = reinterpret_cast<const float4*>(gw)[group];
        const float4 uv = reinterpret_cast<const float4*>(uw)[group];
        gate += xv.x*gv.x + xv.y*gv.y + xv.z*gv.z + xv.w*gv.w;
        up += xv.x*uv.x + xv.y*uv.y + xv.z*uv.z + xv.w*uv.w;
      }
    } else {
      for (int k = lane; k < hidden; k += 32) {
        gate += input[k] * gw[k];
        up += input[k] * uw[k];
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      gate += __shfl_down_sync(0xffffffff, gate, offset);
      up += __shfl_down_sync(0xffffffff, up, offset);
    }
    if (lane != 0) return;
  } else {
    const bool aligned = hidden % 4 == 0 &&
        (reinterpret_cast<uintptr_t>(input) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(gw) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(uw) & 15) == 0;
    if (aligned) {
      for (int group = lane; group < hidden / 4; group += 32) {
        const uint2 xp = reinterpret_cast<const uint2*>(input)[group];
        const uint2 gp = reinterpret_cast<const uint2*>(gw)[group];
        const uint2 upack = reinterpret_cast<const uint2*>(uw)[group];
        const auto* xv = reinterpret_cast<const __nv_bfloat16*>(&xp);
        const auto* gv = reinterpret_cast<const __nv_bfloat16*>(&gp);
        const auto* uv = reinterpret_cast<const __nv_bfloat16*>(&upack);
        for (int j = 0; j < 4; ++j) {
          const float value = to_float(xv[j]);
          gate += value * to_float(gv[j]);
          up += value * to_float(uv[j]);
        }
      }
    } else {
      for (int k = lane; k < hidden; k += 32) {
        const float xv = to_float(input[k]);
        gate += xv * to_float(gw[k]);
        up += xv * to_float(uw[k]);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      gate += __shfl_down_sync(0xffffffff, gate, offset);
      up += __shfl_down_sync(0xffffffff, up, offset);
    }
    if (lane != 0) return;
  }
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    gate = to_float(from_float<T>(gate));
    up = to_float(from_float<T>(up));
  }
  float activated = gate / (1.0f + expf(-gate));
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    activated = to_float(from_float<T>(activated));
  }
  out[index] = from_float<T>(activated * up);
}

__global__ void project_rope_cache_bf16_tensorcore_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* cache, const float* cos_table, const float* sin_table,
    int start_pos, int rows, int n_kv_heads, int head_dim, int hidden) {
  constexpr int TILE = 16, WARPS = 4;
  __shared__ __nv_bfloat16 a_tile[TILE * TILE];
  __shared__ __nv_bfloat16 b_tile[WARPS * TILE * TILE];
  __shared__ float c_tile[WARPS * TILE * TILE];
  __shared__ __nv_bfloat16 projected[TILE * 64];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head = blockIdx.x;
  const int row_base = blockIdx.y * TILE;
  const int n0 = head * head_dim + warp * TILE;
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int k0 = 0; k0 < hidden; k0 += TILE) {
    for (int index = threadIdx.x; index < TILE * TILE; index += blockDim.x) {
      const int r = index / TILE;
      const int kk = index % TILE;
      const int gr = row_base + r;
      const int gk = k0 + kk;
      a_tile[index] = (gr < rows && gk < hidden)
                          ? x[static_cast<size_t>(gr) * hidden + gk]
                          : __float2bfloat16(0.0f);
    }
    for (int index = lane; index < TILE * TILE; index += 32) {
      const int c = index / TILE;
      const int kk = index % TILE;
      const int gk = k0 + kk;
      b_tile[warp * TILE * TILE + index] = (n0 + c < n_kv_heads * head_dim && gk < hidden)
          ? weight[static_cast<size_t>(n0 + c) * hidden + gk]
          : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __nv_bfloat16,
                   wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __nv_bfloat16,
                   wmma::col_major> b;
    wmma::load_matrix_sync(a, a_tile, TILE);
    wmma::load_matrix_sync(b, b_tile + warp * TILE * TILE, TILE);
    wmma::mma_sync(acc, a, b, acc);
    __syncthreads();
  }
  wmma::store_matrix_sync(c_tile + warp * TILE * TILE, acc, TILE,
                          wmma::mem_row_major);
  __syncthreads();
  for (int index = threadIdx.x; index < WARPS * TILE * TILE; index += blockDim.x) {
    const int tile = index / (TILE * TILE);
    const int within = index % (TILE * TILE);
    projected[(within / TILE) * 64 + tile * TILE + within % TILE] =
        __float2bfloat16_rn(c_tile[index]);
  }
  __syncthreads();
  for (int index = threadIdx.x; index < TILE * (head_dim / 2); index += blockDim.x) {
    const int r = index / (head_dim / 2);
    const int i = index % (head_dim / 2);
    const int global_row = row_base + r;
    if (global_row >= rows) continue;
    const int j = i + head_dim / 2;
    const float a = __bfloat162float(projected[r * 64 + i]);
    const float b = __bfloat162float(projected[r * 64 + j]);
    const size_t table_idx = static_cast<size_t>(start_pos + global_row) * (head_dim / 2) + i;
    const float c = cos_table[table_idx];
    const float s = sin_table[table_idx];
    __nv_bfloat16* dst = cache + static_cast<size_t>(global_row) * n_kv_heads * head_dim + head * head_dim;
    dst[i] = __float2bfloat16_rn(a * c - b * s);
    dst[j] = __float2bfloat16_rn(b * c + a * s);
  }
}

__global__ void linear_swiglu_bf16_tensorcore_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight, __nv_bfloat16* out, int rows,
    int intermediate, int hidden) {
  constexpr int TILE = 16, WARPS = 4;
  __shared__ __nv_bfloat16 a_tile[TILE * TILE];
  __shared__ __nv_bfloat16 gate_tile[WARPS * TILE * TILE];
  __shared__ __nv_bfloat16 up_tile[WARPS * TILE * TILE];
  __shared__ float gate_acc[WARPS * TILE * TILE];
  __shared__ float up_acc[WARPS * TILE * TILE];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_base = blockIdx.y * TILE;
  const int n0 = blockIdx.x * WARPS * TILE + warp * TILE;
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> gate_result;
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> up_result;
  wmma::fill_fragment(gate_result, 0.0f);
  wmma::fill_fragment(up_result, 0.0f);
  for (int k0 = 0; k0 < hidden; k0 += TILE) {
    for (int index = threadIdx.x; index < TILE * TILE; index += blockDim.x) {
      const int r = index / TILE;
      const int kk = index % TILE;
      const int gr = row_base + r;
      const int gk = k0 + kk;
      a_tile[index] = (gr < rows && gk < hidden)
                          ? x[static_cast<size_t>(gr) * hidden + gk]
                          : __float2bfloat16(0.0f);
    }
    for (int index = lane; index < TILE * TILE; index += 32) {
      const int c = index / TILE;
      const int kk = index % TILE;
      const int gc = n0 + c;
      const int gk = k0 + kk;
      gate_tile[warp * TILE * TILE + index] = (gc < intermediate && gk < hidden)
          ? gate_weight[static_cast<size_t>(gc) * hidden + gk]
          : __float2bfloat16(0.0f);
      up_tile[warp * TILE * TILE + index] = (gc < intermediate && gk < hidden)
          ? up_weight[static_cast<size_t>(gc) * hidden + gk]
          : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __nv_bfloat16,
                   wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __nv_bfloat16,
                   wmma::col_major> gate_b, up_b;
    wmma::load_matrix_sync(a, a_tile, TILE);
    wmma::load_matrix_sync(gate_b, gate_tile + warp * TILE * TILE, TILE);
    wmma::load_matrix_sync(up_b, up_tile + warp * TILE * TILE, TILE);
    wmma::mma_sync(gate_result, a, gate_b, gate_result);
    wmma::mma_sync(up_result, a, up_b, up_result);
    __syncthreads();
  }
  wmma::store_matrix_sync(gate_acc + warp * TILE * TILE, gate_result, TILE,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(up_acc + warp * TILE * TILE, up_result, TILE,
                          wmma::mem_row_major);
  __syncthreads();
  for (int index = threadIdx.x; index < WARPS * TILE * TILE; index += blockDim.x) {
    const int tile = index / (TILE * TILE);
    const int within = index % (TILE * TILE);
    const int row = row_base + within / TILE;
    const int col = blockIdx.x * WARPS * TILE + tile * TILE + within % TILE;
    if (row >= rows || col >= intermediate) continue;
    float gate = __bfloat162float(__float2bfloat16_rn(gate_acc[index]));
    const float up = __bfloat162float(__float2bfloat16_rn(up_acc[index]));
    float activated = gate / (1.0f + expf(-gate));
    activated = __bfloat162float(__float2bfloat16_rn(activated));
    out[static_cast<size_t>(row) * intermediate + col] =
        __float2bfloat16_rn(activated * up);
  }
}

template <typename T>
__global__ void swiglu_kernel(const T* gate, const T* up, T* out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = to_float(gate[i]);
  float activated = g / (1.0f + expf(-g));
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    activated = to_float(from_float<T>(activated));
  }
  out[i] = from_float<T>(activated * to_float(up[i]));
}

template <typename T, typename OutT>
__global__ void gemm_naive_kernel(const T* x, const T* weight, OutT* out,
                                  int m, int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;
  float acc = 0.0f;
  for (int i = 0; i < k; ++i) {
    acc += to_float(x[static_cast<size_t>(row) * k + i]) *
           to_float(weight[static_cast<size_t>(col) * k + i]);
  }
  out[static_cast<size_t>(row) * n + col] = from_float<OutT>(acc);
}

__global__ void gemm_f32_tiled_kernel(const float* x, const float* weight,
                                     float* out, int m, int n, int k) {
  constexpr int BM = 8, BN = 32, BK = 32;
  __shared__ float a_tile[BM * BK];
  __shared__ float b_tile[BN * BK];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * BM + threadIdx.y;
  const int col = blockIdx.x * BN + threadIdx.x;
  float acc = 0.0f;
  float compensation = 0.0f;
  for (int k0 = 0; k0 < k; k0 += BK) {
    for (int index = tid; index < BM * BK; index += blockDim.x * blockDim.y) {
      const int r = index / BK;
      const int kk = index % BK;
      const int gr = blockIdx.y * BM + r;
      const int gk = k0 + kk;
      a_tile[index] = (gr < m && gk < k) ? x[static_cast<size_t>(gr) * k + gk] : 0.0f;
    }
    for (int index = tid; index < BN * BK; index += blockDim.x * blockDim.y) {
      const int c = index / BK;
      const int kk = index % BK;
      const int gc = blockIdx.x * BN + c;
      const int gk = k0 + kk;
      b_tile[index] = (gc < n && gk < k) ? weight[static_cast<size_t>(gc) * k + gk] : 0.0f;
    }
    __syncthreads();
    if (row < m && col < n) {
      for (int kk = 0; kk < BK; ++kk) {
        const float term = a_tile[threadIdx.y * BK + kk] * b_tile[threadIdx.x * BK + kk];
        const float corrected = term - compensation;
        const float next = acc + corrected;
        compensation = (next - acc) - corrected;
        acc = next;
      }
    }
    __syncthreads();
  }
  if (row < m && col < n) out[static_cast<size_t>(row) * n + col] = acc;
}

__global__ void gemm_bf16_tensorcore_kernel(const __nv_bfloat16* x,
                                            const __nv_bfloat16* weight,
                                            __nv_bfloat16* out, int m, int n,
                                            int k) {
  constexpr int TILE = 16, WARPS = 4;
  __shared__ __nv_bfloat16 a_tile[TILE * TILE];
  __shared__ __nv_bfloat16 b_tile[WARPS * TILE * TILE];
  __shared__ float c_tile[WARPS * TILE * TILE];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int m0 = blockIdx.y * TILE;
  const int n0 = blockIdx.x * WARPS * TILE + warp * TILE;
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k0 = 0; k0 < k; k0 += TILE) {
    for (int index = threadIdx.x; index < TILE * TILE; index += blockDim.x) {
      const int r = index / TILE;
      const int kk = index % TILE;
      const int gr = m0 + r;
      const int gk = k0 + kk;
      a_tile[index] = (gr < m && gk < k) ? x[static_cast<size_t>(gr) * k + gk]
                                                : __float2bfloat16(0.0f);
    }
    for (int index = lane; index < TILE * TILE; index += 32) {
      const int c = index / TILE;
      const int kk = index % TILE;
      const int gc = n0 + c;
      const int gk = k0 + kk;
      b_tile[warp * TILE * TILE + index] =
          (gc < n && gk < k) ? weight[static_cast<size_t>(gc) * k + gk]
                             : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __nv_bfloat16,
                   wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __nv_bfloat16,
                   wmma::col_major> b;
    wmma::load_matrix_sync(a, a_tile, TILE);
    wmma::load_matrix_sync(b, b_tile + warp * TILE * TILE, TILE);
    wmma::mma_sync(acc, a, b, acc);
    __syncthreads();
  }
  wmma::store_matrix_sync(c_tile + warp * TILE * TILE, acc, TILE,
                          wmma::mem_row_major);
  __syncthreads();
  for (int index = threadIdx.x; index < WARPS * TILE * TILE; index += blockDim.x) {
    const int tile = index / (TILE * TILE);
    const int within = index % (TILE * TILE);
    const int row = m0 + within / TILE;
    const int col = blockIdx.x * WARPS * TILE + tile * TILE + within % TILE;
    if (row < m && col < n) {
      out[static_cast<size_t>(row) * n + col] =
          __float2bfloat16_rn(c_tile[index]);
    }
  }
}

__global__ void project_rope_cache_f32_tiled_kernel(
    const float* x, const float* weight, float* cache,
    const float* cos_table, const float* sin_table, int start_pos, int rows,
    int n_kv_heads, int head_dim, int hidden) {
  constexpr int BM = 8, BN = 32, BK = 32;
  __shared__ float a_tile[BM * BK];
  __shared__ float w0_tile[BN * BK];
  __shared__ float w1_tile[BN * BK];
  const int row = blockIdx.y * BM + threadIdx.y;
  const int pair = threadIdx.x;
  const int head = blockIdx.x;
  float a = 0.0f, b = 0.0f, ca = 0.0f, cb = 0.0f;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int k0 = 0; k0 < hidden; k0 += BK) {
    for (int index = tid; index < BM * BK; index += BM * BN) {
      const int r = index / BK, kk = index % BK;
      const int gr = blockIdx.y * BM + r, gk = k0 + kk;
      a_tile[index] = gr < rows && gk < hidden ? x[static_cast<size_t>(gr) * hidden + gk] : 0.0f;
    }
    for (int index = tid; index < BN * BK; index += BM * BN) {
      const int c = index / BK, kk = index % BK, gk = k0 + kk;
      w0_tile[index] = (head * head_dim + c < n_kv_heads * head_dim && gk < hidden)
          ? weight[static_cast<size_t>(head * head_dim + c) * hidden + gk] : 0.0f;
      w1_tile[index] = (head * head_dim + c + head_dim / 2 < n_kv_heads * head_dim && gk < hidden)
          ? weight[static_cast<size_t>(head * head_dim + c + head_dim / 2) * hidden + gk] : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < BK; ++kk) {
      const float xv = a_tile[threadIdx.y * BK + kk];
      const float term_a = xv * w0_tile[threadIdx.x * BK + kk];
      const float ya = term_a - ca, next_a = a + ya;
      ca = (next_a - a) - ya; a = next_a;
      const float term_b = xv * w1_tile[threadIdx.x * BK + kk];
      const float yb = term_b - cb, next_b = b + yb;
      cb = (next_b - b) - yb; b = next_b;
    }
    __syncthreads();
  }
  if (row < rows) {
    const size_t table_idx = static_cast<size_t>(start_pos + row) * (head_dim / 2) + pair;
    const float c = cos_table[table_idx], s = sin_table[table_idx];
    float* dst = cache + static_cast<size_t>(row) * n_kv_heads * head_dim + head * head_dim;
    dst[pair] = a * c - b * s;
    dst[pair + head_dim / 2] = b * c + a * s;
  }
}

__global__ void linear_swiglu_f32_tiled_kernel(
    const float* x, const float* gate_weight, const float* up_weight,
    float* out, int rows, int intermediate, int hidden) {
  constexpr int BM = 8, BN = 32, BK = 32;
  __shared__ float a_tile[BM * BK];
  __shared__ float gate_tile[BN * BK];
  __shared__ float up_tile[BN * BK];
  const int row = blockIdx.y * BM + threadIdx.y;
  const int col = blockIdx.x * BN + threadIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  float gate = 0.0f, up = 0.0f, cg = 0.0f, cu = 0.0f;
  for (int k0 = 0; k0 < hidden; k0 += BK) {
    for (int index = tid; index < BM * BK; index += BM * BN) {
      const int r = index / BK, kk = index % BK;
      const int gr = blockIdx.y * BM + r, gk = k0 + kk;
      a_tile[index] = gr < rows && gk < hidden ? x[static_cast<size_t>(gr) * hidden + gk] : 0.0f;
    }
    for (int index = tid; index < BN * BK; index += BM * BN) {
      const int c = index / BK, kk = index % BK;
      const int gc = blockIdx.x * BN + c, gk = k0 + kk;
      gate_tile[index] = gc < intermediate && gk < hidden
          ? gate_weight[static_cast<size_t>(gc) * hidden + gk] : 0.0f;
      up_tile[index] = gc < intermediate && gk < hidden
          ? up_weight[static_cast<size_t>(gc) * hidden + gk] : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < BK; ++kk) {
      const float xv = a_tile[threadIdx.y * BK + kk];
      const float gt = xv * gate_tile[threadIdx.x * BK + kk];
      const float gy = gt - cg, gn = gate + gy;
      cg = (gn - gate) - gy; gate = gn;
      const float ut = xv * up_tile[threadIdx.x * BK + kk];
      const float uy = ut - cu, un = up + uy;
      cu = (un - up) - uy; up = un;
    }
    __syncthreads();
  }
  if (row < rows && col < intermediate) {
    out[static_cast<size_t>(row) * intermediate + col] =
        (gate / (1.0f + expf(-gate))) * up;
  }
}

template <typename T, typename OutT>
__global__ void gemv_naive_kernel(const T* x, const T* weight, OutT* out,
                                  int n, int k) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= n) return;
  const T* wrow = weight + static_cast<size_t>(row) * k;
  float acc = 0.0f;
  constexpr int values_per_vector = std::is_same<T, float>::value ? 4 : 4;
  const bool aligned = (k % values_per_vector) == 0 &&
      (reinterpret_cast<uintptr_t>(x) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(wrow) & 15) == 0;
  if constexpr (std::is_same<T, float>::value) {
    if (aligned) {
      const float4* xv = reinterpret_cast<const float4*>(x);
      const float4* wv = reinterpret_cast<const float4*>(wrow);
      for (int i = lane; i < k / 4; i += 32) {
        const float4 a = xv[i];
        const float4 b = wv[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
      }
    } else {
      for (int i = lane; i < k; i += 32) acc += x[i] * wrow[i];
    }
  } else if (aligned) {
    const uint2* xv = reinterpret_cast<const uint2*>(x);
    const uint2* wv = reinterpret_cast<const uint2*>(wrow);
    for (int i = lane; i < k / 4; i += 32) {
      const uint2 ap = xv[i];
      const uint2 bp = wv[i];
      const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(&ap);
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&bp);
      acc += to_float(a[0]) * to_float(b[0]);
      acc += to_float(a[1]) * to_float(b[1]);
      acc += to_float(a[2]) * to_float(b[2]);
      acc += to_float(a[3]) * to_float(b[3]);
    }
  } else {
    for (int i = lane; i < k; i += 32) {
      acc += to_float(x[i]) * to_float(wrow[i]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  if (lane == 0) out[row] = from_float<OutT>(acc);
}

template <typename T>
__global__ void rope_kernel(T* q, T* k, const float* cos_table,
                            const float* sin_table, int start_pos, int rows,
                            int n_heads, int n_kv_heads, int head_dim) {
  const int token = blockIdx.x;
  if (token >= rows) return;
  const int half = head_dim / 2;
  const int all_heads = n_heads + n_kv_heads;
  const int position = start_pos + token;
  for (int pair = threadIdx.x; pair < all_heads * half; pair += blockDim.x) {
    const int head = pair / half;
    const int i = pair % half;
    const int j = i + half;
    T* row = head < n_heads
                 ? q + (static_cast<size_t>(token) * n_heads + head) * head_dim
                 : k + (static_cast<size_t>(token) * n_kv_heads + head - n_heads) * head_dim;
    const float a = to_float(row[i]);
    const float b = to_float(row[j]);
    const size_t table_idx = static_cast<size_t>(position) * half + i;
    const float c = cos_table[table_idx];
    const float s = sin_table[table_idx];
    row[i] = from_float<T>(a * c - b * s);
    row[j] = from_float<T>(b * c + a * s);
  }
}

template <typename T>
__global__ void attention_naive_kernel(const T* q, const T* kcache,
                                       const T* vcache, T* out,
                                       int start_pos, int rows, int n_heads,
                                       int n_kv_heads, int head_dim,
                                       float scale) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int kv_head = head / (n_heads / n_kv_heads);
  const int len = start_pos + token + 1;
  extern __shared__ float scores[];
  const T* qrow = q + (static_cast<size_t>(token) * n_heads + head) * head_dim;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int n_warps = blockDim.x >> 5;

  for (int pos = warp; pos < len; pos += n_warps) {
    const T* krow = kcache + (static_cast<size_t>(pos) * n_kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int i = lane; i < head_dim; i += 32) {
      dot += to_float(qrow[i]) * to_float(krow[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
    if (lane == 0) scores[pos] = dot * scale;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (int pos = 0; pos < len; ++pos) maximum = fmaxf(maximum, scores[pos]);
    float denominator = 0.0f;
    for (int pos = 0; pos < len; ++pos) {
      scores[pos] = expf(scores[pos] - maximum);
      denominator += scores[pos];
    }
    for (int pos = 0; pos < len; ++pos) scores[pos] /= denominator;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float value = 0.0f;
    for (int pos = 0; pos < len; ++pos) {
      const T* vrow = vcache + (static_cast<size_t>(pos) * n_kv_heads + kv_head) * head_dim;
      value += scores[pos] * to_float(vrow[i]);
    }
    out[(static_cast<size_t>(token) * n_heads + head) * head_dim + i] =
        from_float<T>(value);
  }
}

__global__ void argmax_kernel(const float* logits, int* result, int n) {
  __shared__ float values[1024];
  __shared__ int indices[1024];
  float best = -INFINITY;
  int best_index = INT_MAX;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float value = logits[i];
    if (!isnan(value) &&
        (value > best || (value == best && i < best_index))) {
      best = value;
      best_index = i;
    }
  }
  values[threadIdx.x] = best;
  indices[threadIdx.x] = best_index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other = values[threadIdx.x + stride];
      const int other_index = indices[threadIdx.x + stride];
      if (other > values[threadIdx.x] ||
          (other == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
        values[threadIdx.x] = other;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) result[0] = indices[0];
}

extern "C" int engine_residual_f32(const float* a, const float* b, float* out,
                                   size_t n, cudaStream_t stream) {
  if (n == 0) return 0;
  residual_add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(a, b, out, n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int engine_residual_bf16(const __nv_bfloat16* a,
                                    const __nv_bfloat16* b,
                                    __nv_bfloat16* out, size_t n,
                                    cudaStream_t stream) {
  if (n == 0) return 0;
  residual_add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(a, b, out, n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int engine_residual_rmsnorm_f32(
    const float* x, const float* proj, const float* weight, float* residual,
    float* normalized, int rows, int hidden, float eps, cudaStream_t stream) {
  if (!x || !proj || !weight || !residual || !normalized || rows <= 0 || hidden <= 0)
    return int(cudaErrorInvalidValue);
  residual_rmsnorm_kernel<<<rows, 256, 0, stream>>>(
      x, proj, weight, residual, normalized, rows, hidden, eps);
  return int(cudaGetLastError());
}

extern "C" int engine_residual_rmsnorm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* proj,
    const __nv_bfloat16* weight, __nv_bfloat16* residual,
    __nv_bfloat16* normalized, int rows, int hidden, float eps,
    cudaStream_t stream) {
  if (!x || !proj || !weight || !residual || !normalized || rows <= 0 || hidden <= 0)
    return int(cudaErrorInvalidValue);
  residual_rmsnorm_kernel<<<rows, 256, 0, stream>>>(
      x, proj, weight, residual, normalized, rows, hidden, eps);
  return int(cudaGetLastError());
}

extern "C" int engine_project_rope_cache_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* cache, const float* cos_table, const float* sin_table,
    int start_pos, int rows, int n_kv_heads, int head_dim, int hidden,
    cudaStream_t stream) {
  if (!x || !weight || !cache || !cos_table || !sin_table || start_pos < 0 ||
      rows <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 || hidden <= 0)
    return int(cudaErrorInvalidValue);
  const int pairs = rows * n_kv_heads * (head_dim / 2);
  if (rows >= 16 && head_dim == 64) {
    project_rope_cache_bf16_tensorcore_kernel<<<
        dim3(n_kv_heads, (rows + 15) / 16), 128, 0, stream>>>(
        x, weight, cache, cos_table, sin_table, start_pos, rows, n_kv_heads,
        head_dim, hidden);
  } else {
    project_rope_cache_kernel<<<(pairs + 7) / 8, 256, 0, stream>>>(
        x, weight, cache, cos_table, sin_table, start_pos, rows, n_kv_heads,
        head_dim, hidden);
  }
  return int(cudaGetLastError());
}

extern "C" int engine_project_rope_cache_f32(
    const float* x, const float* weight, float* cache, const float* cos_table,
    const float* sin_table, int start_pos, int rows, int n_kv_heads,
    int head_dim, int hidden, cudaStream_t stream) {
  if (!x || !weight || !cache || !cos_table || !sin_table || start_pos < 0 ||
      rows <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 || hidden <= 0)
    return int(cudaErrorInvalidValue);
  const int pairs = rows * n_kv_heads * (head_dim / 2);
  if (rows > 1 && head_dim == 64) {
    project_rope_cache_f32_tiled_kernel<<<
        dim3(n_kv_heads, (rows + 7) / 8), dim3(32, 8), 0, stream>>>(
        x, weight, cache, cos_table, sin_table, start_pos, rows, n_kv_heads,
        head_dim, hidden);
  } else {
    project_rope_cache_kernel<<<(pairs + 7) / 8, 256, 0, stream>>>(
        x, weight, cache, cos_table, sin_table, start_pos, rows, n_kv_heads,
        head_dim, hidden);
  }
  return int(cudaGetLastError());
}

extern "C" int engine_linear_swiglu_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight, __nv_bfloat16* out, int rows,
    int intermediate, int hidden, cudaStream_t stream) {
  if (!x || !gate_weight || !up_weight || !out || rows <= 0 ||
      intermediate <= 0 || hidden <= 0)
    return int(cudaErrorInvalidValue);
  const size_t count = static_cast<size_t>(rows) * intermediate;
  if (rows >= 16) {
    linear_swiglu_bf16_tensorcore_kernel<<<
        dim3((intermediate + 63) / 64, (rows + 15) / 16), 128, 0, stream>>>(
        x, gate_weight, up_weight, out, rows, intermediate, hidden);
  } else {
    linear_swiglu_kernel<<<(count + 7) / 8, 256, 0, stream>>>(
        x, gate_weight, up_weight, out, rows, intermediate, hidden);
  }
  return int(cudaGetLastError());
}

extern "C" int engine_linear_swiglu_f32(
    const float* x, const float* gate_weight, const float* up_weight,
    float* out, int rows, int intermediate, int hidden, cudaStream_t stream) {
  if (!x || !gate_weight || !up_weight || !out || rows <= 0 ||
      intermediate <= 0 || hidden <= 0)
    return int(cudaErrorInvalidValue);
  const size_t count = static_cast<size_t>(rows) * intermediate;
  if (rows > 1) {
    linear_swiglu_f32_tiled_kernel<<<
        dim3((intermediate + 31) / 32, (rows + 7) / 8), dim3(32, 8), 0, stream>>>(
        x, gate_weight, up_weight, out, rows, intermediate, hidden);
  } else {
    linear_swiglu_kernel<<<(count + 7) / 8, 256, 0, stream>>>(
        x, gate_weight, up_weight, out, rows, intermediate, hidden);
  }
  return int(cudaGetLastError());
}

extern "C" int engine_embedding_f32(const float* table, const int* ids,
                                     float* out, int rows, int hidden,
                                     cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) return int(cudaErrorInvalidValue);
  embedding_kernel<<<rows, 256, 0, stream>>>(table, ids, out, rows, hidden);
  return int(cudaGetLastError());
}

extern "C" int engine_embedding_bf16(const __nv_bfloat16* table, const int* ids,
                                      __nv_bfloat16* out, int rows, int hidden,
                                      cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) return int(cudaErrorInvalidValue);
  embedding_kernel<<<rows, 256, 0, stream>>>(table, ids, out, rows, hidden);
  return int(cudaGetLastError());
}

extern "C" int engine_rmsnorm_f32(const float* x, const float* weight,
                                  float* out, int rows, int hidden, float eps,
                                  cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) return int(cudaErrorInvalidValue);
  rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, weight, out, rows, hidden, eps);
  return int(cudaGetLastError());
}

extern "C" int engine_rmsnorm_bf16(const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* out, int rows, int hidden,
                                   float eps, cudaStream_t stream) {
  if (rows <= 0 || hidden <= 0) return int(cudaErrorInvalidValue);
  rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, weight, out, rows, hidden, eps);
  return int(cudaGetLastError());
}

extern "C" int engine_swiglu_f32(const float* gate, const float* up,
                                 float* out, size_t n, cudaStream_t stream) {
  if (n == 0) return 0;
  swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
  return int(cudaGetLastError());
}

extern "C" int engine_swiglu_bf16(const __nv_bfloat16* gate,
                                  const __nv_bfloat16* up,
                                  __nv_bfloat16* out, size_t n,
                                  cudaStream_t stream) {
  if (n == 0) return 0;
  swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
  return int(cudaGetLastError());
}

extern "C" int engine_gemm_f32(const float* x, const float* weight, float* out,
                               int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return int(cudaErrorInvalidValue);
  const dim3 block(32, 8);
  const dim3 grid((n + 31) / 32, (m + 7) / 8);
  gemm_f32_tiled_kernel<<<grid, block, 0, stream>>>(x, weight, out, m, n, k);
  return int(cudaGetLastError());
}

extern "C" int engine_gemm_bf16(const __nv_bfloat16* x,
                                const __nv_bfloat16* weight,
                                __nv_bfloat16* out, int m, int n, int k,
                                cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return int(cudaErrorInvalidValue);
  const dim3 block(128);
  const dim3 grid((n + 63) / 64, (m + 15) / 16);
  gemm_bf16_tensorcore_kernel<<<grid, block, 0, stream>>>(x, weight, out, m, n, k);
  return int(cudaGetLastError());
}

extern "C" int engine_gemv_f32(const float* x, const float* weight, float* out,
                               int n, int k, cudaStream_t stream) {
  if (n <= 0 || k <= 0) return int(cudaErrorInvalidValue);
  gemv_naive_kernel<<<(n + 7) / 8, 256, 0, stream>>>(x, weight, out, n, k);
  return int(cudaGetLastError());
}

extern "C" int engine_gemv_bf16(const __nv_bfloat16* x,
                                const __nv_bfloat16* weight,
                                __nv_bfloat16* out, int n, int k,
                                cudaStream_t stream) {
  if (n <= 0 || k <= 0) return int(cudaErrorInvalidValue);
  gemv_naive_kernel<<<(n + 7) / 8, 256, 0, stream>>>(x, weight, out, n, k);
  return int(cudaGetLastError());
}

extern "C" int engine_gemv_bf16_f32(const __nv_bfloat16* x,
                                    const __nv_bfloat16* weight, float* out,
                                    int n, int k, cudaStream_t stream) {
  if (n <= 0 || k <= 0) return int(cudaErrorInvalidValue);
  gemv_naive_kernel<<<(n + 7) / 8, 256, 0, stream>>>(x, weight, out, n, k);
  return int(cudaGetLastError());
}

extern "C" int engine_build_rope_tables(float* cos_out, float* sin_out,
                                        int max_seq, int head_dim, float theta,
                                        float factor, float low_freq_factor,
                                        float high_freq_factor,
                                        int original_max_position_embeddings) {
  if (!cos_out || !sin_out || max_seq <= 0 || head_dim <= 0 || head_dim % 2 ||
      theta <= 0.0f || factor <= 0.0f || low_freq_factor <= 0.0f ||
      high_freq_factor < low_freq_factor || original_max_position_embeddings <= 0) {
    return int(cudaErrorInvalidValue);
  }
  const int half = head_dim / 2;
  const double low_wavelength = static_cast<double>(original_max_position_embeddings) / low_freq_factor;
  const double high_wavelength = static_cast<double>(original_max_position_embeddings) / high_freq_factor;
  for (int i = 0; i < half; ++i) {
    const double inv = 1.0 / std::pow(static_cast<double>(theta), 2.0 * i / head_dim);
    const double wavelength = 6.283185307179586476925286766559 / inv;
    double scaled_inv = inv;
    if (wavelength > low_wavelength) {
      scaled_inv = inv / factor;
    } else if (wavelength >= high_wavelength) {
      const double smooth = (original_max_position_embeddings / wavelength - low_freq_factor) /
                            (high_freq_factor - low_freq_factor);
      scaled_inv = (1.0 - smooth) * (inv / factor) + smooth * inv;
    }
    for (int pos = 0; pos < max_seq; ++pos) {
      const double angle = static_cast<double>(pos) * scaled_inv;
      const size_t idx = static_cast<size_t>(pos) * half + i;
      cos_out[idx] = static_cast<float>(std::cos(angle));
      sin_out[idx] = static_cast<float>(std::sin(angle));
    }
  }
  return 0;
}

extern "C" int engine_rope_f32(float* q, float* k, const float* cos_table,
                               const float* sin_table, int start_pos, int rows,
                               int n_heads, int n_kv_heads, int head_dim,
                               cudaStream_t stream) {
  if (!q || (!k && n_kv_heads > 0) || !cos_table || !sin_table || start_pos < 0 || rows <= 0 ||
      n_heads <= 0 || n_kv_heads < 0 || head_dim <= 0 || head_dim % 2) {
    return int(cudaErrorInvalidValue);
  }
  rope_kernel<<<rows, 256, 0, stream>>>(q, k, cos_table, sin_table, start_pos,
                                        rows, n_heads, n_kv_heads, head_dim);
  return int(cudaGetLastError());
}

extern "C" int engine_rope_bf16(__nv_bfloat16* q, __nv_bfloat16* k,
                                const float* cos_table, const float* sin_table,
                                int start_pos, int rows, int n_heads,
                                int n_kv_heads, int head_dim,
                                cudaStream_t stream) {
  if (!q || (!k && n_kv_heads > 0) || !cos_table || !sin_table || start_pos < 0 || rows <= 0 ||
      n_heads <= 0 || n_kv_heads < 0 || head_dim <= 0 || head_dim % 2) {
    return int(cudaErrorInvalidValue);
  }
  rope_kernel<<<rows, 256, 0, stream>>>(q, k, cos_table, sin_table, start_pos,
                                        rows, n_heads, n_kv_heads, head_dim);
  return int(cudaGetLastError());
}

extern "C" int engine_attention_f32(const float* q, const float* kcache,
                                    const float* vcache, float* out,
                                    int start_pos, int rows, int n_heads,
                                    int n_kv_heads, int head_dim, float scale,
                                    cudaStream_t stream) {
  if (!q || !kcache || !vcache || !out || start_pos < 0 || rows <= 0 ||
      n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0 ||
      head_dim <= 0 || head_dim % 32 != 0 || start_pos + rows > 12000) {
    return int(cudaErrorInvalidValue);
  }
  const dim3 grid(n_heads, rows);
  const size_t shared_bytes = static_cast<size_t>(start_pos + rows) * sizeof(float);
  attention_naive_kernel<<<grid, 128, shared_bytes, stream>>>(
      q, kcache, vcache, out, start_pos, rows, n_heads, n_kv_heads, head_dim, scale);
  return int(cudaGetLastError());
}

extern "C" int engine_attention_bf16(const __nv_bfloat16* q,
                                     const __nv_bfloat16* kcache,
                                     const __nv_bfloat16* vcache,
                                     __nv_bfloat16* out, int start_pos,
                                     int rows, int n_heads, int n_kv_heads,
                                     int head_dim, float scale,
                                     cudaStream_t stream) {
  if (!q || !kcache || !vcache || !out || start_pos < 0 || rows <= 0 ||
      n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0 ||
      head_dim <= 0 || head_dim % 32 != 0 || start_pos + rows > 12000) {
    return int(cudaErrorInvalidValue);
  }
  const dim3 grid(n_heads, rows);
  const size_t shared_bytes = static_cast<size_t>(start_pos + rows) * sizeof(float);
  attention_naive_kernel<<<grid, 128, shared_bytes, stream>>>(
      q, kcache, vcache, out, start_pos, rows, n_heads, n_kv_heads, head_dim, scale);
  return int(cudaGetLastError());
}

extern "C" int engine_argmax_f32(const float* logits, int* index, int n,
                                 cudaStream_t stream) {
  if (!logits || !index || n <= 0) return int(cudaErrorInvalidValue);
  argmax_kernel<<<1, 1024, 0, stream>>>(logits, index, n);
  return int(cudaGetLastError());
}
