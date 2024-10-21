# llama-3-korean-bllossom-8b-q4-docker
8GB GPU로 8B 모델 추론을 추구하면 안되는걸까

## 환경 구축

```bash
git clone https://github.com/developer0hye/llama-3-korean-bllossom-8b-q4-docker.git
cd llama-3-korean-bllossom-8b-q4-docker
docker build -t llama-bllossom:latest .
docker run --gpus all -it -v ./:/workspace --ipc=host --name bllossom llama-bllossom:latest
python bllossom.py 
```

## 출력
```
/opt/conda/lib/python3.11/site-packages/huggingface_hub/file_download.py:797: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
llama_model_loader: loaded meta data with 30 key-value pairs and 291 tensors from /ckpt/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = llama-3-Korean-Bllossom-8B
llama_model_loader: - kv   2:                          llama.block_count u32              = 32
llama_model_loader: - kv   3:                       llama.context_length u32              = 8192
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   7:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   8:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 15
llama_model_loader: - kv  11:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  12:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  14:                         tokenizer.ggml.pre str              = llama-bpe
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  16:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  17:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  18:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  19:                tokenizer.ggml.eos_token_id u32              = 128009
llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 128001
llama_model_loader: - kv  21:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                                general.url str              = https://huggingface.co/mradermacher/l...
llama_model_loader: - kv  24:              mradermacher.quantize_version str              = 2
llama_model_loader: - kv  25:                  mradermacher.quantized_by str              = mradermacher
llama_model_loader: - kv  26:                  mradermacher.quantized_at str              = 2024-06-18T22:17:22+02:00
llama_model_loader: - kv  27:                  mradermacher.quantized_on str              = leia
llama_model_loader: - kv  28:                         general.source.url str              = https://huggingface.co/MLP-KTLim/llam...
llama_model_loader: - kv  29:                  mradermacher.convert_type str              = hf
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:  193 tensors
llama_model_loader: - type q6_K:   33 tensors
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 0.8000 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 8192
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 8192
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 8.03 B
llm_load_print_meta: model size       = 4.58 GiB (4.89 BPW) 
llm_load_print_meta: general.name     = llama-3-Korean-Bllossom-8B
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: PAD token        = 128001 '<|end_of_text|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3070, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.27 MiB
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:        CPU buffer size =   281.81 MiB
llm_load_tensors:      CUDA0 buffer size =  4403.49 MiB
........................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =    64.00 MiB
llama_new_context_with_model: KV self size  =   64.00 MiB, K (f16):   32.00 MiB, V (f16):   32.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   258.50 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =     9.01 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 2
AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | RISCV_VECT = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
Model metadata: {'mradermacher.quantized_on': 'leia', 'mradermacher.quantize_version': '2', 'tokenizer.chat_template': "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}", 'tokenizer.ggml.padding_token_id': '128001', 'tokenizer.ggml.eos_token_id': '128009', 'general.quantization_version': '2', 'tokenizer.ggml.model': 'gpt2', 'mradermacher.convert_type': 'hf', 'general.url': 'https://huggingface.co/mradermacher/llama-3-Korean-Bllossom-8B-GGUF', 'general.architecture': 'llama', 'mradermacher.quantized_by': 'mradermacher', 'llama.rope.freq_base': '500000.000000', 'mradermacher.quantized_at': '2024-06-18T22:17:22+02:00', 'tokenizer.ggml.pre': 'llama-bpe', 'llama.context_length': '8192', 'general.name': 'llama-3-Korean-Bllossom-8B', 'llama.embedding_length': '4096', 'llama.feed_forward_length': '14336', 'general.source.url': 'https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B', 'llama.attention.layer_norm_rms_epsilon': '0.000010', 'tokenizer.ggml.bos_token_id': '128000', 'llama.attention.head_count': '32', 'llama.block_count': '32', 'llama.attention.head_count_kv': '8', 'general.file_type': '15', 'llama.vocab_size': '128256', 'llama.rope.dimension_count': '128'}
Available chat formats from metadata: chat_template.default
Using gguf chat template: {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>

' }}{% endif %}
Using chat eos_token: <|eot_id|>
Using chat bos_token: <|begin_of_text|>
/opt/conda/lib/python3.11/site-packages/llama_cpp/llama.py:1238: RuntimeWarning: Detected duplicate leading "<|begin_of_text|>" in prompt, this will likely reduce response quality, consider removing it...
  warnings.warn(
llama_perf_context_print:        load time =     118.84 ms
llama_perf_context_print: prompt eval time =       0.00 ms /    72 tokens (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:        eval time =       0.00 ms /    51 runs   (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:       total time =     919.07 ms /   123 tokens
root@1c97fed4267b:/workspace# python bllossom.py 
/opt/conda/lib/python3.11/site-packages/huggingface_hub/file_download.py:797: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
llama_model_loader: loaded meta data with 30 key-value pairs and 291 tensors from /ckpt/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = llama-3-Korean-Bllossom-8B
llama_model_loader: - kv   2:                          llama.block_count u32              = 32
llama_model_loader: - kv   3:                       llama.context_length u32              = 8192
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   7:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   8:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 15
llama_model_loader: - kv  11:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  12:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  14:                         tokenizer.ggml.pre str              = llama-bpe
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  16:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  17:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  18:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  19:                tokenizer.ggml.eos_token_id u32              = 128009
llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 128001
llama_model_loader: - kv  21:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                                general.url str              = https://huggingface.co/mradermacher/l...
llama_model_loader: - kv  24:              mradermacher.quantize_version str              = 2
llama_model_loader: - kv  25:                  mradermacher.quantized_by str              = mradermacher
llama_model_loader: - kv  26:                  mradermacher.quantized_at str              = 2024-06-18T22:17:22+02:00
llama_model_loader: - kv  27:                  mradermacher.quantized_on str              = leia
llama_model_loader: - kv  28:                         general.source.url str              = https://huggingface.co/MLP-KTLim/llam...
llama_model_loader: - kv  29:                  mradermacher.convert_type str              = hf
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:  193 tensors
llama_model_loader: - type q6_K:   33 tensors
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 0.8000 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 8192
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 8192
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 8.03 B
llm_load_print_meta: model size       = 4.58 GiB (4.89 BPW) 
llm_load_print_meta: general.name     = llama-3-Korean-Bllossom-8B
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: PAD token        = 128001 '<|end_of_text|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3070, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.27 MiB
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:        CPU buffer size =   281.81 MiB
llm_load_tensors:      CUDA0 buffer size =  4403.49 MiB
........................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =    64.00 MiB
llama_new_context_with_model: KV self size  =   64.00 MiB, K (f16):   32.00 MiB, V (f16):   32.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   258.50 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =     9.01 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 2
AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | RISCV_VECT = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
Model metadata: {'mradermacher.quantized_on': 'leia', 'mradermacher.quantize_version': '2', 'tokenizer.chat_template': "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}", 'tokenizer.ggml.padding_token_id': '128001', 'tokenizer.ggml.eos_token_id': '128009', 'general.quantization_version': '2', 'tokenizer.ggml.model': 'gpt2', 'mradermacher.convert_type': 'hf', 'general.url': 'https://huggingface.co/mradermacher/llama-3-Korean-Bllossom-8B-GGUF', 'general.architecture': 'llama', 'mradermacher.quantized_by': 'mradermacher', 'llama.rope.freq_base': '500000.000000', 'mradermacher.quantized_at': '2024-06-18T22:17:22+02:00', 'tokenizer.ggml.pre': 'llama-bpe', 'llama.context_length': '8192', 'general.name': 'llama-3-Korean-Bllossom-8B', 'llama.embedding_length': '4096', 'llama.feed_forward_length': '14336', 'general.source.url': 'https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B', 'llama.attention.layer_norm_rms_epsilon': '0.000010', 'tokenizer.ggml.bos_token_id': '128000', 'llama.attention.head_count': '32', 'llama.block_count': '32', 'llama.attention.head_count_kv': '8', 'general.file_type': '15', 'llama.vocab_size': '128256', 'llama.rope.dimension_count': '128'}
Available chat formats from metadata: chat_template.default
Using gguf chat template: {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>

' }}{% endif %}
Using chat eos_token: <|eot_id|>
Using chat bos_token: <|begin_of_text|>
/opt/conda/lib/python3.11/site-packages/llama_cpp/llama.py:1238: RuntimeWarning: Detected duplicate leading "<|begin_of_text|>" in prompt, this will likely reduce response quality, consider removing it...
  warnings.warn(
llama_perf_context_print:        load time =     120.07 ms
llama_perf_context_print: prompt eval time =       0.00 ms /    72 tokens (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:        eval time =       0.00 ms /    51 runs   (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:       total time =     939.60 ms /   123 tokens
안녕하세요! 저는 유용한 AI 어시스턴트입니다. 여러분의 질문에 답변을 제공하고, 정보를 제공하며, 다양한 주제에 대해 도움을 드릴 수 있습니다. 어떤 도움이 필요하신가요?
```

## GPU 상태

### 실행 전
![실행-전-gpu-status](<실행-전-gpu-status.png>)

### 실행 후
![실행-후-gpu-status](<실행-후-gpu-status.png>)

## 결론
추구해도 됩니다.

## 참고 자료

https://huggingface.co/QuantFactory/llama-3-Korean-Bllossom-8B-GGUF

https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M
