#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>

extern "C" int engine_residual_f32(const float* a, const float* b, float* out,
                                   size_t n, cudaStream_t stream);
extern "C" int engine_residual_bf16(const __nv_bfloat16* a,
                                    const __nv_bfloat16* b,
                                    __nv_bfloat16* out, size_t n,
                                    cudaStream_t stream);
extern "C" int engine_residual_rmsnorm_f32(const float* x, const float* proj,
                                           const float* weight, float* residual,
                                           float* normalized, int rows,
                                           int hidden, float eps,
                                           cudaStream_t stream);
extern "C" int engine_residual_rmsnorm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* proj,
    const __nv_bfloat16* weight, __nv_bfloat16* residual,
    __nv_bfloat16* normalized, int rows, int hidden, float eps,
    cudaStream_t stream);
extern "C" int engine_project_rope_cache_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* cache, const float* cos_table, const float* sin_table,
    int start_pos, int rows, int n_kv_heads, int head_dim, int hidden,
    cudaStream_t stream);
extern "C" int engine_project_rope_cache_f32(
    const float* x, const float* weight, float* cache, const float* cos_table,
    const float* sin_table, int start_pos, int rows, int n_kv_heads,
    int head_dim, int hidden, cudaStream_t stream);
extern "C" int engine_linear_swiglu_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight, __nv_bfloat16* out, int rows,
    int intermediate, int hidden, cudaStream_t stream);
extern "C" int engine_linear_swiglu_f32(
    const float* x, const float* gate_weight, const float* up_weight,
    float* out, int rows, int intermediate, int hidden, cudaStream_t stream);

extern "C" int engine_embedding_f32(const float* table, const int* ids,
                                     float* out, int rows, int hidden,
                                     cudaStream_t stream);
extern "C" int engine_embedding_bf16(const __nv_bfloat16* table, const int* ids,
                                      __nv_bfloat16* out, int rows, int hidden,
                                      cudaStream_t stream);
extern "C" int engine_rmsnorm_f32(const float* x, const float* weight,
                                  float* out, int rows, int hidden, float eps,
                                  cudaStream_t stream);
extern "C" int engine_rmsnorm_bf16(const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* out, int rows, int hidden,
                                   float eps, cudaStream_t stream);
extern "C" int engine_swiglu_f32(const float* gate, const float* up,
                                 float* out, size_t n, cudaStream_t stream);
extern "C" int engine_swiglu_bf16(const __nv_bfloat16* gate,
                                  const __nv_bfloat16* up,
                                  __nv_bfloat16* out, size_t n,
                                  cudaStream_t stream);

extern "C" int engine_gemm_f32(const float* x, const float* weight, float* out,
                               int m, int n, int k, cudaStream_t stream);
extern "C" int engine_gemm_bf16(const __nv_bfloat16* x,
                                const __nv_bfloat16* weight,
                                __nv_bfloat16* out, int m, int n, int k,
                                cudaStream_t stream);
extern "C" int engine_gemv_f32(const float* x, const float* weight, float* out,
                               int n, int k, cudaStream_t stream);
extern "C" int engine_gemv_bf16(const __nv_bfloat16* x,
                                const __nv_bfloat16* weight,
                                __nv_bfloat16* out, int n, int k,
                                cudaStream_t stream);
extern "C" int engine_gemv_bf16_f32(const __nv_bfloat16* x,
                                    const __nv_bfloat16* weight, float* out,
                                    int n, int k, cudaStream_t stream);
extern "C" int engine_build_rope_tables(float* cos_out, float* sin_out,
                                        int max_seq, int head_dim, float theta,
                                        float factor, float low_freq_factor,
                                        float high_freq_factor,
                                        int original_max_position_embeddings);
extern "C" int engine_rope_f32(float* q, float* k, const float* cos_table,
                               const float* sin_table, int start_pos, int rows,
                               int n_heads, int n_kv_heads, int head_dim,
                               cudaStream_t stream);
extern "C" int engine_rope_bf16(__nv_bfloat16* q, __nv_bfloat16* k,
                                const float* cos_table, const float* sin_table,
                                int start_pos, int rows, int n_heads,
                                int n_kv_heads, int head_dim,
                                cudaStream_t stream);
extern "C" int engine_attention_f32(const float* q, const float* kcache,
                                    const float* vcache, float* out,
                                    int start_pos, int rows, int n_heads,
                                    int n_kv_heads, int head_dim, float scale,
                                    cudaStream_t stream);
extern "C" int engine_attention_bf16(const __nv_bfloat16* q,
                                     const __nv_bfloat16* kcache,
                                     const __nv_bfloat16* vcache,
                                     __nv_bfloat16* out, int start_pos,
                                     int rows, int n_heads, int n_kv_heads,
                                     int head_dim, float scale,
                                     cudaStream_t stream);
extern "C" int engine_argmax_f32(const float* logits, int* index, int n,
                                 cudaStream_t stream);
