import json
import threading
import ollama
from tqdm import tqdm

# Fixed system prompt
SYSTEM_PROMPT = """Write the following program for CUDA with c and the output file should be a .cu file.
Make sure to only output the .cu file and not any other supporting information, text, or so on.
Make sure to write the entire code from the start to the end.
Before you write the code, write in a multiline comment the aim of the program which is passed in this prompt word to word. Along with that, add your entire thinking to that comment as well. Only after that, are you supposed to start responding with the actual code outside of the comment.
"""

THREAD_COUNT = 4

class ModelInstance:
    """Persistent Ollama model instance."""
    def __init__(self):
        self.model_name = "gpt-oss:20b"
        print(f"[INFO] Initialized model {self.model_name}")

    def query(self, user_prompt):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response["message"]["content"]

def process_prompt(model_instance, tag: str, prompt: str):
    """Query the model instance and save output to a .cu file."""
    output = model_instance.query(prompt)
    filename = f"{tag}.cu"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)

def worker(items, position):
    """Thread worker: initializes its own model instance and processes prompts with a progress bar."""
    model_instance = ModelInstance()
    pbar = tqdm(total=len(items), desc=f"Thread {position+1}", position=position)
    for tag, prompt in items:
        process_prompt(model_instance, tag, prompt)
        pbar.update(1)
    pbar.close()

def main():
    # Load prompts
    with open("prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    items = list(prompts.items())
    if len(items) == 0:
        print("[ERROR] No prompts found in prompts.json")
        return

    # Split items into THREAD_COUNT chunks
    chunks = [items[i::THREAD_COUNT] for i in range(THREAD_COUNT)]
    threads = []

    for i, chunk in enumerate(chunks):
        t = threading.Thread(target=worker, args=(chunk, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("[✔] All prompts processed.")

if __name__ == "__main__":
    main()
