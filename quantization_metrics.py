import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import matplotlib.pyplot as plt

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

INPUT_TEXT = "Artificial intelligence (AI) is the capability of a computer system or machine to mimic human intelligence, including learning, reasoning, problem-solving, and decision-making. A subset of AI where computers use data and statistical algorithms to learn patterns and make predictions without being explicitly programmed.An advanced branch of ML that uses multi-layered artificial neural networks to mimic the human brain for complex tasks like speech and image recognition."

def load_model_and_tokenizer(model_name: str):
  """
  Loads the model and tokenizer for the given model name.
  Returns the model and tokenizer.
  """
  float_point_type = torch.float16
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device == "cuda": # Corrected from "gpu" to "cuda"
    float_point_type = torch.float16 # Using float16 for GPU, can be changed to float32 if needed
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
  model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=float_point_type)
  if device == "cuda": # Corrected from "gpu" to "cuda"
    # Tokenizer is usually CPU-bound, so no need to move it to GPU
    model.to(torch.device(device))
  model.eval()
  return model, tokenizer

def tokenize_with_labels(tokenizer, input_text):
  """
  Tokenizes the input text and returns the tokens and the length of the tokens.
  """
  tokens = tokenizer(input_text, return_tensors="pt")
  tokens['labels'] = tokens['input_ids'].clone()
  tokens_len = tokens['labels'].shape[1]
  return tokens, tokens_len

def compute_peak_memory_loss(model, inputs, device):
  """
    Computes the peak memory and loss for the given model and inputs.
  """
  device = str(device)
  if device == "cuda":
    "Reset the peak memory usage"
    torch.cuda.reset_peak_memory_stats()
  """
    Forward pass through the model to compute the loss and also memory.
  """
  with torch.no_grad():
    output = model(**inputs)
  if device == "cuda":
    "GPU will wait until the operation is completed"
    torch.cuda.synchronize()
    "Get the peak memory usage and the loss"
    return torch.cuda.max_memory_allocated() / 1024**2 , output.loss.item()
  else:
    return 0.0, output.loss.item()

def compute_perplexity(loss):
  """
    Computes the perplexity for the given loss.
  """
  return math.exp(loss)

def time_forward(model, inputs, device, num_warmup: int = 1, num_runs: int = 3):
  """
    Times the forward pass of the model for the given inputs.
  """
  device = str(device)
  "Warm up the model"
  with torch.no_grad():
    for _ in range(num_warmup):
      _ = model(**inputs)
  "Time the forward pass"
  if device == "cuda":
    torch.cuda.synchronize()
  latencies = []
  with torch.no_grad():
    for _ in range(num_runs):
      start = time.perf_counter()
      _ = model(**inputs)
      if device == "cuda":
        torch.cuda.synchronize()
      end_time = time.perf_counter()
      latencies.append(end_time - start)
  return sum(latencies) / max(len(latencies), 1)

def sweep_context_length(
    tokenizer, model, device,
    targets=(64, 256, 1024),
    base_text=None, build_text_fn=None,
    warmup=1, runs=3, plot=True,
):
    # 1) model's max context, leave room for special tokens
    max_len = getattr(model.config, "max_position_embeddings", 2048) or 2048
    targets_used = [t for t in targets if t <= max_len - 2]
    if not targets_used:
        raise ValueError(f"No targets fit within model max context ({max_len}).")

    # 2) base paragraph
    if base_text is None:
        base_text = INPUT_TEXT
    if not base_text:
        raise ValueError("No base_text provided and select_text() returned empty.")

    # 3) default builder: repeat base_text until it reaches target_tokens
    def _default_builder(tok, text, target_tokens):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            raise ValueError("base_text tokenizes to zero tokens.")
        reps = (target_tokens // len(ids)) + 1
        long_ids = (ids * reps)[:target_tokens]
        return tok.decode(long_ids)
    builder = build_text_fn or _default_builder

    lengths, peak_list, lat_list = [], [], []

    # 4) run each length
    for target in targets_used:
        text = builder(tokenizer, base_text, target)
        inputs, input_len = tokenize_with_labels(tokenizer, text)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        peak_mib, _loss = compute_peak_memory_loss(model, inputs, device)
        latency_s = time_forward(model, inputs, device, num_warmup=warmup, num_runs=runs)

        lengths.append(input_len)
        peak_list.append(peak_mib)
        lat_list.append(latency_s)
        print(f"[ctx] tokens={input_len:5d} | peak={peak_mib:8.1f} MiB | fwd={latency_s:.4f}s")

    # 5) plots — one figure each, markers + grid, no custom colors
    if plot:
        plt.figure(); plt.plot(lengths, peak_list, marker="o")
        plt.xlabel("input tokens"); plt.ylabel("peak memory (MiB)")
        plt.title("Peak memory vs context length"); plt.grid(True); plt.show()

        plt.figure(); plt.plot(lengths, lat_list, marker="o")
        plt.xlabel("input tokens"); plt.ylabel("forward latency (s)")
        plt.title("Latency vs context length"); plt.grid(True); plt.show()

    # 6) results
    return {
        "lengths": lengths,
        "peak_mib": peak_list,
        "latency_s": lat_list,
        "max_ctx": max_len,
        "targets_used": targets_used,
    }

def load_with_dtype(dtype_str: str):
    dtype = torch.float16 if dtype_str == "float16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device).eval()
    return tok, mod, device

def compare_precision_memory_perplexity(
    model_name, tokenizer, load_with_dtype_fn,
    precisions=("float16", "float32"),
    base_text=None,
):
    if base_text is None:
        base_text = INPUT_TEXT
    inputs_s, Ls = tokenize_with_labels(tokenizer, base_text)
    results = {}
    for precision in precisions:
        tok_p, mod_p, dev_p = load_with_dtype_fn(precision)
        inp = {k: v.to(dev_p) for k, v in inputs_s.items()}
        peak_mib, loss_val = compute_peak_memory_loss(mod_p, inp, dev_p)
        ppl = compute_perplexity(loss_val)
        results[precision] = {"peak_mib": peak_mib, "ppl": ppl}
        print(f"[prec] {precision:8s} | peak={peak_mib:8.1f} MiB | ppl={ppl:.3f}")
    return results

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

sweep_context_length(tokenizer, model, "cuda")

compare_precision_memory_perplexity(MODEL_NAME, tokenizer, load_with_dtype)
