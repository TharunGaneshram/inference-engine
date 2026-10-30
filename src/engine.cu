#include "inference/engine.h"
#include "inference/kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
thread_local std::string last_error;

void check_cuda(cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(call) check_cuda((call), #call)

void check_launch(int status, const char* operation) {
  check_cuda(static_cast<cudaError_t>(status), operation);
}

struct StreamOwner {
  cudaStream_t stream = nullptr;
  void create() { CUDA_CHECK(cudaStreamCreate(&stream)); }
  ~StreamOwner() { if (stream) cudaStreamDestroy(stream); }
  StreamOwner() = default;
  StreamOwner(const StreamOwner&) = delete;
  StreamOwner& operator=(const StreamOwner&) = delete;
};

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) : bytes(n) {
    if (n != 0) CUDA_CHECK(cudaMalloc(&ptr, n));
  }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr(std::exchange(other.ptr, nullptr)), bytes(std::exchange(other.bytes, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr) cudaFree(ptr);
      ptr = std::exchange(other.ptr, nullptr);
      bytes = std::exchange(other.bytes, 0);
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct TensorSlot { size_t offset, bytes; bool uploaded = false; };

struct Workspace {
  DeviceBuffer ids, x, xn, q, attn, proj, gate, up, logits, next_token;
  int max_tokens = 0;
};

struct Engine {
  StreamOwner stream;  // first member, so the stream is destroyed after all buffers
  EngineConfig cfg;
  DeviceBuffer arena;
  std::unordered_map<std::string, TensorSlot> slots;
  DeviceBuffer kv_k, kv_v, rope_cos, rope_sin;
  Workspace ws;
  int kv_len = 0;
  int last_rows = 0;
  bool finalized = false;

  explicit Engine(const EngineConfig& config) : cfg(config) {
    if (cfg.n_layers <= 0 || cfg.hidden <= 0 || cfg.n_heads <= 0 ||
        cfg.n_kv_heads <= 0 || cfg.head_dim <= 0 || cfg.intermediate <= 0 ||
        cfg.vocab <= 0 || cfg.max_seq <= 0) {
      throw std::runtime_error("engine_create: model dimensions must be positive");
    }
    if (cfg.hidden != cfg.n_heads * cfg.head_dim) {
      throw std::runtime_error("engine_create: hidden must equal n_heads * head_dim");
    }
    if (cfg.n_heads % cfg.n_kv_heads != 0) {
      throw std::runtime_error("engine_create: n_heads must be divisible by n_kv_heads");
    }
    if (cfg.max_seq > 12000 || cfg.head_dim % 2 != 0) {
      throw std::runtime_error("engine_create: unsupported max_seq or head_dim");
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice(0)");
    stream.create();

    size_t total = 0;
    auto add = [&](const std::string& name, size_t elements) {
      total = (total + 255) & ~size_t(255);
      const size_t bytes = elements * sizeof(uint16_t);
      slots.emplace(name, TensorSlot{total, bytes, false});
      total += bytes;
    };
    add("model.embed_tokens.weight", size_t(cfg.vocab) * cfg.hidden);
    for (int l = 0; l < cfg.n_layers; ++l) {
      const std::string p = "model.layers." + std::to_string(l) + ".";
      add(p + "input_layernorm.weight", cfg.hidden);
      add(p + "self_attn.q_proj.weight", size_t(cfg.n_heads) * cfg.head_dim * cfg.hidden);
      add(p + "self_attn.k_proj.weight", size_t(cfg.n_kv_heads) * cfg.head_dim * cfg.hidden);
      add(p + "self_attn.v_proj.weight", size_t(cfg.n_kv_heads) * cfg.head_dim * cfg.hidden);
      add(p + "self_attn.o_proj.weight", size_t(cfg.hidden) * cfg.n_heads * cfg.head_dim);
      add(p + "post_attention_layernorm.weight", cfg.hidden);
      add(p + "mlp.gate_proj.weight", size_t(cfg.intermediate) * cfg.hidden);
      add(p + "mlp.up_proj.weight", size_t(cfg.intermediate) * cfg.hidden);
      add(p + "mlp.down_proj.weight", size_t(cfg.hidden) * cfg.intermediate);
    }
    add("model.norm.weight", cfg.hidden);
    if (!cfg.tie_embeddings) add("lm_head.weight", size_t(cfg.vocab) * cfg.hidden);
    arena = DeviceBuffer(total);

    const size_t kv_elements = size_t(cfg.n_layers) * cfg.max_seq * cfg.n_kv_heads * cfg.head_dim;
    kv_k = DeviceBuffer(kv_elements * sizeof(uint16_t));
    kv_v = DeviceBuffer(kv_elements * sizeof(uint16_t));
    const size_t rope_elements = size_t(cfg.max_seq) * (cfg.head_dim / 2);
    rope_cos = DeviceBuffer(rope_elements * sizeof(float));
    rope_sin = DeviceBuffer(rope_elements * sizeof(float));
    std::vector<float> cos_host(rope_elements), sin_host(rope_elements);
    const int table_status = engine_build_rope_tables(
        cos_host.data(), sin_host.data(), cfg.max_seq, cfg.head_dim, cfg.rope_theta,
        cfg.rope_factor, cfg.rope_low_freq_factor, cfg.rope_high_freq_factor,
        cfg.rope_original_max_position_embeddings);
    check_launch(table_status, "engine_build_rope_tables");
    check_cuda(cudaMemcpy(rope_cos.ptr, cos_host.data(), rope_elements * sizeof(float), cudaMemcpyHostToDevice),
               "copy RoPE cosine table");
    check_cuda(cudaMemcpy(rope_sin.ptr, sin_host.data(), rope_elements * sizeof(float), cudaMemcpyHostToDevice),
               "copy RoPE sine table");

    ws.max_tokens = 2048;
    const size_t tokens = ws.max_tokens;
    ws.ids = DeviceBuffer(tokens * sizeof(int));
    ws.x = DeviceBuffer(tokens * cfg.hidden * sizeof(uint16_t));
    ws.xn = DeviceBuffer(tokens * cfg.hidden * sizeof(uint16_t));
    ws.q = DeviceBuffer(tokens * cfg.hidden * sizeof(uint16_t));
    ws.attn = DeviceBuffer(tokens * cfg.hidden * sizeof(uint16_t));
    ws.proj = DeviceBuffer(tokens * cfg.hidden * sizeof(uint16_t));
    ws.gate = DeviceBuffer(tokens * cfg.intermediate * sizeof(uint16_t));
    ws.up = DeviceBuffer(tokens * cfg.intermediate * sizeof(uint16_t));
    ws.logits = DeviceBuffer(size_t(cfg.vocab) * sizeof(float));
    ws.next_token = DeviceBuffer(sizeof(int));
  }

  void upload(const char* name_ptr, const void* host, size_t nbytes) {
    if (finalized) throw std::runtime_error("engine_upload_tensor: engine is finalized");
    if (!name_ptr) throw std::runtime_error("engine_upload_tensor: tensor name is null");
    const std::string name(name_ptr);
    const auto it = slots.find(name);
    if (it == slots.end()) throw std::runtime_error("engine_upload_tensor: unknown tensor '" + name + "'");
    if (nbytes != it->second.bytes) {
      std::ostringstream msg;
      msg << "engine_upload_tensor: tensor '" << name << "' has " << nbytes
          << " bytes; expected " << it->second.bytes;
      throw std::runtime_error(msg.str());
    }
    if (!host) throw std::runtime_error("engine_upload_tensor: host pointer is null for '" + name + "'");
    check_cuda(cudaMemcpy(static_cast<char*>(arena.ptr) + it->second.offset,
                          host, nbytes, cudaMemcpyHostToDevice),
               "cudaMemcpy tensor '" + name + "'");
    it->second.uploaded = true;
  }

  void finalize() {
    for (const auto& entry : slots) {
      if (!entry.second.uploaded) {
        throw std::runtime_error("engine_finalize: missing tensor '" + entry.first + "'");
      }
    }
    finalized = true;
  }

  void* tensor(const std::string& name) {
    const auto it = slots.find(name);
    if (it == slots.end()) {
      if (name == "lm_head.weight" && cfg.tie_embeddings) return tensor("model.embed_tokens.weight");
      throw std::runtime_error("missing tensor view '" + name + "'");
    }
    return static_cast<char*>(arena.ptr) + it->second.offset;
  }

  void* layer_tensor(int layer, const char* suffix) {
    return tensor("model.layers." + std::to_string(layer) + "." + suffix);
  }

  void linear(const __nv_bfloat16* x, const __nv_bfloat16* weight,
              __nv_bfloat16* out, int m, int n, int k) {
    if (m == 1) {
      check_launch(engine_gemv_bf16(x, weight, out, n, k, stream.stream), "GEMV");
    } else {
      check_launch(engine_gemm_bf16(x, weight, out, m, n, k, stream.stream), "GEMM");
    }
  }

  int forward(const int* ids_host, int m, int max_layers = -1,
              bool want_token = true) {
    if (!finalized) throw std::runtime_error("engine_forward: weights are not finalized");
    if (!ids_host || m <= 0 || m > ws.max_tokens) {
      throw std::runtime_error("engine_forward: token count must be between 1 and 2048");
    }
    if (kv_len + m > cfg.max_seq) throw std::runtime_error("engine_forward: prompt exceeds max_seq");
    const int n_run = max_layers < 0 ? cfg.n_layers : max_layers;
    if (n_run < 0 || n_run > cfg.n_layers) throw std::runtime_error("engine_forward: max_layers is out of range");
    for (int i = 0; i < m; ++i) {
      if (ids_host[i] < 0 || ids_host[i] >= cfg.vocab) {
        throw std::runtime_error("engine_forward: token id is outside the vocabulary");
      }
    }
    const int p = kv_len;
    last_rows = m;
    check_cuda(cudaMemcpyAsync(ws.ids.ptr, ids_host, size_t(m) * sizeof(int),
                               cudaMemcpyHostToDevice, stream.stream), "copy input token ids");
    check_launch(engine_embedding_bf16(
        static_cast<const __nv_bfloat16*>(tensor("model.embed_tokens.weight")),
        static_cast<const int*>(ws.ids.ptr), static_cast<__nv_bfloat16*>(ws.x.ptr),
        m, cfg.hidden, stream.stream), "embedding");

    const size_t kv_layer_bytes = size_t(cfg.max_seq) * cfg.n_kv_heads * cfg.head_dim * sizeof(uint16_t);
    for (int l = 0; l < n_run; ++l) {
      auto* x = static_cast<__nv_bfloat16*>(ws.x.ptr);
      auto* xn = static_cast<__nv_bfloat16*>(ws.xn.ptr);
      auto* q = static_cast<__nv_bfloat16*>(ws.q.ptr);
      auto* attn = static_cast<__nv_bfloat16*>(ws.attn.ptr);
      auto* proj = static_cast<__nv_bfloat16*>(ws.proj.ptr);
      auto* kc = reinterpret_cast<__nv_bfloat16*>(static_cast<char*>(kv_k.ptr) + l * kv_layer_bytes) + size_t(p) * cfg.n_kv_heads * cfg.head_dim;
      auto* vc = reinterpret_cast<__nv_bfloat16*>(static_cast<char*>(kv_v.ptr) + l * kv_layer_bytes) + size_t(p) * cfg.n_kv_heads * cfg.head_dim;
      check_launch(engine_rmsnorm_bf16(x, static_cast<const __nv_bfloat16*>(layer_tensor(l, "input_layernorm.weight")), xn, m, cfg.hidden, cfg.rms_eps, stream.stream), "attention RMSNorm");
      linear(xn, static_cast<const __nv_bfloat16*>(layer_tensor(l, "self_attn.q_proj.weight")), q, m, cfg.hidden, cfg.hidden);
      linear(xn, static_cast<const __nv_bfloat16*>(layer_tensor(l, "self_attn.k_proj.weight")), kc, m, cfg.n_kv_heads * cfg.head_dim, cfg.hidden);
      linear(xn, static_cast<const __nv_bfloat16*>(layer_tensor(l, "self_attn.v_proj.weight")), vc, m, cfg.n_kv_heads * cfg.head_dim, cfg.hidden);
      check_launch(engine_rope_bf16(q, kc, static_cast<const float*>(rope_cos.ptr), static_cast<const float*>(rope_sin.ptr), p, m, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, stream.stream), "RoPE");
      auto* kbase = reinterpret_cast<const __nv_bfloat16*>(static_cast<char*>(kv_k.ptr) + l * kv_layer_bytes);
      auto* vbase = reinterpret_cast<const __nv_bfloat16*>(static_cast<char*>(kv_v.ptr) + l * kv_layer_bytes);
      check_launch(engine_attention_bf16(q, kbase, vbase, attn, p, m, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, 1.0f / std::sqrt(static_cast<float>(cfg.head_dim)), stream.stream), "attention");
      linear(attn, static_cast<const __nv_bfloat16*>(layer_tensor(l, "self_attn.o_proj.weight")), proj, m, cfg.hidden, cfg.hidden);
      check_launch(engine_residual_bf16(x, proj, x, size_t(m) * cfg.hidden, stream.stream), "attention residual add");
      check_launch(engine_rmsnorm_bf16(x, static_cast<const __nv_bfloat16*>(layer_tensor(l, "post_attention_layernorm.weight")), xn, m, cfg.hidden, cfg.rms_eps, stream.stream), "MLP RMSNorm");
      auto* gate = static_cast<__nv_bfloat16*>(ws.gate.ptr);
      auto* up = static_cast<__nv_bfloat16*>(ws.up.ptr);
      linear(xn, static_cast<const __nv_bfloat16*>(layer_tensor(l, "mlp.gate_proj.weight")), gate, m, cfg.intermediate, cfg.hidden);
      linear(xn, static_cast<const __nv_bfloat16*>(layer_tensor(l, "mlp.up_proj.weight")), up, m, cfg.intermediate, cfg.hidden);
      check_launch(engine_swiglu_bf16(gate, up, gate, size_t(m) * cfg.intermediate, stream.stream), "SwiGLU");
      linear(gate, static_cast<const __nv_bfloat16*>(layer_tensor(l, "mlp.down_proj.weight")), proj, m, cfg.hidden, cfg.intermediate);
      check_launch(engine_residual_bf16(x, proj, x, size_t(m) * cfg.hidden, stream.stream), "MLP residual add");
    }

    if (max_layers >= 0 || !want_token) {
      if (max_layers < 0) kv_len += m;
      check_cuda(cudaStreamSynchronize(stream.stream), "synchronize forward layers");
      return -1;
    }
    auto* last_x = static_cast<__nv_bfloat16*>(ws.x.ptr) + size_t(m - 1) * cfg.hidden;
    check_launch(engine_rmsnorm_bf16(last_x, static_cast<const __nv_bfloat16*>(tensor("model.norm.weight")), static_cast<__nv_bfloat16*>(ws.xn.ptr), 1, cfg.hidden, cfg.rms_eps, stream.stream), "final RMSNorm");
    check_launch(engine_gemv_bf16_f32(static_cast<const __nv_bfloat16*>(ws.xn.ptr), static_cast<const __nv_bfloat16*>(tensor("lm_head.weight")), static_cast<float*>(ws.logits.ptr), cfg.vocab, cfg.hidden, stream.stream), "lm_head GEMV");
    check_launch(engine_argmax_f32(static_cast<const float*>(ws.logits.ptr), static_cast<int*>(ws.next_token.ptr), cfg.vocab, stream.stream), "argmax");
    int token = -1;
    check_cuda(cudaMemcpyAsync(&token, ws.next_token.ptr, sizeof(token), cudaMemcpyDeviceToHost, stream.stream), "copy next token");
    check_cuda(cudaStreamSynchronize(stream.stream), "synchronize generated token");
    kv_len += m;
    return token;
  }

  int prefill(const int* ids, int n) {
    if (n > ws.max_tokens) throw std::runtime_error("engine_prefill: prompt exceeds workspace; use engine_generate for chunking");
    return forward(ids, n);
  }

  int decode(int token) { return forward(&token, 1); }

  std::vector<int> generate(const int* prompt, int n, int max_new) {
    if (!prompt || n <= 0) throw std::runtime_error("engine_generate: prompt must contain at least one token");
    if (n > cfg.max_seq) throw std::runtime_error("engine_generate: prompt exceeds max_seq");
    if (max_new < 0) throw std::runtime_error("engine_generate: max_new must be nonnegative");
    reset();
    if (max_new == 0) return {};
    int token = -1;
    for (int offset = 0; offset < n;) {
      const int count = std::min(ws.max_tokens, n - offset);
      const bool last = offset + count == n;
      token = forward(prompt + offset, count, -1, last);
      offset += count;
    }
    std::vector<int> output{token};
    while (static_cast<int>(output.size()) < max_new && token != cfg.eos_token_id && kv_len < cfg.max_seq) {
      token = decode(token);
      output.push_back(token);
    }
    return output;
  }

  void reset() { kv_len = 0; }

  void read_residual(void* host, int n) {
    if (!host || n != last_rows * cfg.hidden) throw std::runtime_error("engine_read_residual: output size must equal last call rows * hidden");
    check_cuda(cudaMemcpy(host, ws.x.ptr, size_t(n) * sizeof(uint16_t), cudaMemcpyDeviceToHost), "copy residual stream");
  }

  void read_logits(float* host) {
    if (!host) throw std::runtime_error("engine_last_logits: output pointer is null");
    check_cuda(cudaMemcpy(host, ws.logits.ptr, size_t(cfg.vocab) * sizeof(float), cudaMemcpyDeviceToHost), "copy logits");
  }
};

template <typename F>
int api_call(F&& f) noexcept {
  try {
    f();
    last_error.clear();
    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
  } catch (...) {
    last_error = "unknown C++ exception";
  }
  return -1;
}

template <typename F>
int api_value(F&& f) noexcept {
  try {
    const int value = f();
    last_error.clear();
    return value;
  } catch (const std::exception& e) {
    last_error = e.what();
  } catch (...) {
    last_error = "unknown C++ exception";
  }
  return -1;
}
}  // namespace

extern "C" void* engine_create(const EngineConfig* cfg) {
  try {
    if (!cfg) throw std::runtime_error("engine_create: config is null");
    auto* engine = new Engine(*cfg);
    last_error.clear();
    return engine;
  } catch (const std::exception& e) {
    last_error = e.what();
  } catch (...) {
    last_error = "unknown C++ exception";
  }
  return nullptr;
}

extern "C" int engine_upload_tensor(void* engine, const char* name,
                                    const void* host, size_t nbytes) {
  return api_call([&] {
    if (!engine) throw std::runtime_error("engine_upload_tensor: engine is null");
    static_cast<Engine*>(engine)->upload(name, host, nbytes);
  });
}

extern "C" int engine_finalize(void* engine) {
  return api_call([&] {
    if (!engine) throw std::runtime_error("engine_finalize: engine is null");
    static_cast<Engine*>(engine)->finalize();
  });
}

extern "C" int engine_prefill(void* engine, const int* ids, int n) {
  return api_value([&] {
    if (!engine) throw std::runtime_error("engine_prefill: engine is null");
    return static_cast<Engine*>(engine)->prefill(ids, n);
  });
}

extern "C" int engine_decode(void* engine, int token) {
  return api_value([&] {
    if (!engine) throw std::runtime_error("engine_decode: engine is null");
    return static_cast<Engine*>(engine)->decode(token);
  });
}

extern "C" int engine_generate(void* engine, const int* prompt, int n,
                               int max_new, int* output_tokens) {
  return api_value([&] {
    if (!engine) throw std::runtime_error("engine_generate: engine is null");
    if (max_new > 0 && !output_tokens) throw std::runtime_error("engine_generate: output pointer is null");
    const auto tokens = static_cast<Engine*>(engine)->generate(prompt, n, max_new);
    std::copy(tokens.begin(), tokens.end(), output_tokens);
    return static_cast<int>(tokens.size());
  });
}

extern "C" int engine_reset(void* engine) {
  return api_call([&] {
    if (!engine) throw std::runtime_error("engine_reset: engine is null");
    static_cast<Engine*>(engine)->reset();
  });
}

extern "C" int engine_forward(void* engine, const int* ids, int n, int max_layers) {
  return api_value([&] {
    if (!engine) throw std::runtime_error("engine_forward: engine is null");
    return static_cast<Engine*>(engine)->forward(ids, n, max_layers);
  });
}

extern "C" int engine_read_residual(void* engine, void* host_out, int n) {
  return api_call([&] {
    if (!engine) throw std::runtime_error("engine_read_residual: engine is null");
    static_cast<Engine*>(engine)->read_residual(host_out, n);
  });
}

extern "C" int engine_last_logits(void* engine, float* host_out) {
  return api_call([&] {
    if (!engine) throw std::runtime_error("engine_last_logits: engine is null");
    static_cast<Engine*>(engine)->read_logits(host_out);
  });
}

extern "C" const char* engine_last_error() { return last_error.c_str(); }

extern "C" void engine_destroy(void* engine) { delete static_cast<Engine*>(engine); }
