#!/usr/bin/env python3
"""
run_parallel_ollama.py
- Distributes prompts across GPUs (one process per GPU).
- Each process sets OLLAMA_GPU before importing/using ollama to force per-process GPU binding.
- Expects prompts.json in the same directory with format: { "tag1": "prompt1", "tag2": "prompt2", ... }
"""

import os
import json
import subprocess
from multiprocessing import Process
from math import ceil
from tqdm import tqdm
from typing import List, Tuple

# -------------------------
# Cleaned, strict system prompt
# -------------------------
# This tells the model to output EXACTLY the requested format.
SYSTEM_PROMPT = """You are to generate a complete CUDA program in C. Your output MUST follow this exact format and nothing else:

# Aim: `<<user prompt here in one line>>`

\"\"\" 
<<insert your complete internal reasoning and planning here as a multiline comment>>
\"\"\"

<<insert the full, complete CUDA program code here (start-to-finish), suitable to save as a .cu file>>
 
Rules:
- Output ONLY the three parts above (Aim, triple-quoted reasoning, then the code). No extra text, no explanations, no headings.
- The Aim line must exactly repeat the user's prompt inside backticks, single-line.
- The triple-quoted block must contain your full internal thinking/planning before the code.
- The code must be a complete, compilable CUDA C file.
"""

# -------------------------
# Utilities
# -------------------------
DEFAULT_THREAD_COUNT = 4  # fallback if GPU detection fails

def detect_gpu_count() -> int:
    """Try to detect GPUs using nvidia-smi. Return detected count or DEFAULT_THREAD_COUNT."""
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, text=True)
        lines = [l for l in out.splitlines() if l.strip()]
        n = len(lines)
        if n >= 1:
            return n
    except Exception:
        # nvidia-smi not available or error
        pass
    return DEFAULT_THREAD_COUNT

def chunk_items(items: List[Tuple[str, str]], n_chunks: int) -> List[List[Tuple[str, str]]]:
    """Round-robin split items into n_chunks."""
    if n_chunks <= 0:
        return [items]
    chunks = [[] for _ in range(n_chunks)]
    for i, it in enumerate(items):
        chunks[i % n_chunks].append(it)
    return chunks

# -------------------------
# Worker process
# -------------------------
def worker_process(items: List[Tuple[str, str]], gpu_id: int, system_prompt: str):
    """
    Runs in its own process:
    - Sets OLLAMA_GPU env var (so the underlying runtime binds to specified GPU).
    - Imports ollama after env var is set to avoid driver selection race.
    - Processes each (tag, prompt) pair and writes <tag>.cu.
    """
    # Bind this process to the chosen GPU
    os.environ["OLLAMA_GPU"] = str(gpu_id)
    # Optional: also set CUDA_VISIBLE_DEVICES for libraries depending on it
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import here to ensure the env var is set before any ollama init logic runs.
    import ollama

    model_name = "gpt-oss:20b"
    # Simple progress bar for this process (position = gpu_id helps separate bars)
    pbar = tqdm(total=len(items), desc=f"GPU {gpu_id}", position=gpu_id, leave=True)

    for tag, user_prompt in items:
        # Prepare messages enforcing Aim line: we'll instruct the model with the system prompt and
        # give the user prompt so the Aim line gets the exact prompt text.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Use ollama.chat (synchronous). Depending on ollama version,
            # the return may be streaming or non-streaming. Adjust if needed.
            resp = ollama.chat(model=model_name, messages=messages)
            # Try to extract content robustly
            content = None
            if isinstance(resp, dict):
                # typical shape: {"message": {"content": "<text>"}}
                content = resp.get("message", {}).get("content")
            if content is None:
                # fallback: try stringifying resp
                content = str(resp)
        except Exception as e:
            content = f"/* ERROR: model call failed: {e} */\n"

        # Save output as <tag>.cu
        filename = f"{tag}.cu"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed to write {filename}: {e}")

        pbar.update(1)

    pbar.close()

# -------------------------
# Main: load prompts, spawn processes
# -------------------------
def main():
    # Load prompts
    prompts_path = "prompts.json"
    if not os.path.exists(prompts_path):
        print(f"[ERROR] {prompts_path} not found.")
        return

    with open(prompts_path, "r", encoding="utf-8") as fh:
        prompts = json.load(fh)

    items = list(prompts.items())
    if not items:
        print("[ERROR] No prompts found in prompts.json")
        return

    detected_gpus = detect_gpu_count()
    # Use up to detected_gpus (but not more than number of prompts)
    process_count = min(detected_gpus, len(items))
    if process_count <= 0:
        process_count = DEFAULT_THREAD_COUNT

    print(f"[INFO] Detected GPUs: {detected_gpus}. Launching {process_count} worker processes.")

    chunks = chunk_items(items, process_count)

    processes: List[Process] = []
    for gpu_id, chunk in enumerate(chunks):
        p = Process(target=worker_process, args=(chunk, gpu_id, SYSTEM_PROMPT), daemon=False)
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("[✔] All prompts processed across GPUs.")

if __name__ == "__main__":
    main()
