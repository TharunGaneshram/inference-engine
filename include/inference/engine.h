#pragma once

#include <cstddef>
#include <cstdint>

struct EngineConfig {
  int n_layers, hidden, n_heads, n_kv_heads, head_dim, intermediate, vocab;
  float rms_eps, rope_theta;
  float rope_factor, rope_low_freq_factor, rope_high_freq_factor;
  int rope_original_max_position_embeddings;
  bool tie_embeddings;
  int max_seq;
  int eos_token_id;
};

extern "C" {
void* engine_create(const EngineConfig* cfg);
int engine_upload_tensor(void* engine, const char* name, const void* host,
                         size_t nbytes);
int engine_finalize(void* engine);
int engine_prefill(void* engine, const int* ids, int n);
int engine_decode(void* engine, int token);
int engine_generate(void* engine, const int* prompt, int n, int max_new,
                    int* output_tokens);
int engine_reset(void* engine);
int engine_forward(void* engine, const int* ids, int n, int max_layers);
int engine_read_residual(void* engine, void* host_out, int n);
int engine_last_logits(void* engine, float* host_out);
const char* engine_last_error();
void engine_destroy(void* engine);
}
