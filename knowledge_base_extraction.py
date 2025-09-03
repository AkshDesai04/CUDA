import os
import json
import ollama
import faiss
import numpy as np
import PyPDF2
import time
import sys
import multiprocessing

# --- Configuration ---
CONFIG = {
    "MODEL_NAME": "gpt-oss:20b",
    "EMBED_MODEL_NAME": "mxbai-embed-large", # A high-quality embedding model available on Ollama
    "FILES_DIR": "./files",
    "OUTPUT_DIR": "./output",
    "CHUNK_SIZE": 500,  # words
    "CHUNK_OVERLAP": 50, # words
    "FAISS_K": 4, # Number of relevant chunks to retrieve for answering
    "NUM_GPUS": 3, # Number of GPUs to use for parallel processing
}

class QAGenerator:
    """
    A class to generate Question-Answer pairs from PDF documents using a RAG pipeline.
    """
    def __init__(self, config):
        self.config = config
        # Client initialization is lightweight and safe here.
        # The actual model loading is handled by the Ollama service.
        self.client = ollama.Client()
        self._check_models()
        os.makedirs(self.config["OUTPUT_DIR"], exist_ok=True)
        # We don't print "initialized" here anymore, as it's done per-process.

    def _check_models(self):
        """
        Ensures the required models are available in Ollama.
        This is run by each process, but the pull operation is idempotent.
        """
        try:
            # Safely get the list of models to prevent KeyErrors
            ollama_response = self.client.list()
            models_list = ollama_response.get('models', [])
            models_available = [m.get('name') for m in models_list if m and m.get('name')]

            if f"{self.config['MODEL_NAME']}:latest" not in models_available:
                print(f"[Process {os.getpid()}] Model '{self.config['MODEL_NAME']}' not found. Pulling it now...")
                self.client.pull(self.config['MODEL_NAME'])
            if f"{self.config['EMBED_MODEL_NAME']}:latest" not in models_available:
                print(f"[Process {os.getpid()}] Embedding model '{self.config['EMBED_MODEL_NAME']}' not found. Pulling it now...")
                self.client.pull(self.config['EMBED_MODEL_NAME'])
        except Exception as e:
            print(f"Error connecting to Ollama or pulling models: {e}")
            print("Please ensure Ollama is running and accessible.")
            # Exit the specific worker process, not the whole application
            sys.exit(1)

    def _load_and_chunk_pdf(self, pdf_path):
        """Loads text from a PDF and splits it into chunks."""
        print(f"[{os.getpid()}] Loading and chunking '{os.path.basename(pdf_path)}'...")
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.config["CHUNK_SIZE"] - self.config["CHUNK_OVERLAP"]):
                chunk = " ".join(words[i:i + self.config["CHUNK_SIZE"]])
                chunks.append(chunk)
            print(f"[{os.getpid()}] Document split into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            print(f"[{os.getpid()}] Error reading or chunking PDF {pdf_path}: {e}")
            return []

    def _get_embeddings(self, texts):
        """Generates embeddings for a list of text chunks."""
        print(f"[{os.getpid()}] Generating embeddings for {len(texts)} chunks...")
        embeddings = []
        for i, text in enumerate(texts):
            try:
                response = self.client.embeddings(model=self.config["EMBED_MODEL_NAME"], prompt=text)
                embeddings.append(response["embedding"])
                print(f"  - [{os.getpid()}] Embedded chunk {i+1}/{len(texts)}", end='\r')
            except Exception as e:
                print(f"\n[{os.getpid()}] Error generating embedding for chunk {i+1}: {e}")
                embeddings.append([0.0] * 1024)
        print(f"\n[{os.getpid()}] Embedding generation complete.")
        return np.array(embeddings).astype('float32')

    def _create_faiss_index(self, embeddings):
        """Creates a FAISS index from embeddings."""
        if embeddings.shape[0] == 0:
            return None
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"[{os.getpid()}] FAISS index created with {index.ntotal} vectors.")
        return index

    def _search_faiss_index(self, index, query_embedding):
        """Searches the FAISS index."""
        distances, indices = index.search(np.array([query_embedding]).astype('float32'), self.config["FAISS_K"])
        return indices[0]

    def _generate_response_with_retry(self, model, prompt, max_retries=3):
        """Generates a response from the LLM with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.generate(model=model, prompt=prompt, stream=False)
                return response['response'].strip()
            except Exception as e:
                print(f"\n[{os.getpid()}] LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5 * (attempt + 1))
        return None

    def process_document(self, pdf_path):
        """Main processing pipeline for a single PDF document."""
        doc_name = os.path.basename(pdf_path)
        output_filename = os.path.splitext(doc_name)[0] + ".json"
        output_path = os.path.join(self.config["OUTPUT_DIR"], output_filename)

        print(f"\n{'='*25} [PID:{os.getpid()}] Processing: {doc_name} {'='*25}")

        chunks = self._load_and_chunk_pdf(pdf_path)
        if not chunks:
            return
            
        embeddings = self._get_embeddings(chunks)
        faiss_index = self._create_faiss_index(embeddings)
        if faiss_index is None:
            print(f"[{os.getpid()}] Failed to create FAISS index. Skipping document.")
            return

        qa_pairs = []
        total_chunks = len(chunks)
        print(f"[{os.getpid()}] Starting Q&A generation for {total_chunks} chunks...")

        for i, chunk in enumerate(chunks):
            print(f"\n--- [{os.getpid()}] Processing Chunk {i+1}/{total_chunks} for {doc_name} ---")
            
            question_prompt = f"Based ONLY on the following text, generate one single, clear, and specific question. Do not answer it. Output only the question.\n\nText:\n---\n{chunk}\n---\n\nQuestion:"
            question = self._generate_response_with_retry(self.config["MODEL_NAME"], question_prompt)

            if not question:
                print(f"[{os.getpid()}] Failed to generate question for this chunk. Skipping.")
                continue
            
            print(f"  [{os.getpid()}] [Generated Question]: {question}")

            query_embedding_response = self.client.embeddings(model=self.config["EMBED_MODEL_NAME"], prompt=question)
            query_embedding = query_embedding_response["embedding"]
            
            retrieved_indices = self._search_faiss_index(faiss_index, query_embedding)
            context = "\n---\n".join(chunks[idx] for idx in retrieved_indices)

            answer_prompt = f"You are an expert Q&A system. Use the provided context to give a comprehensive and accurate answer to the question. If the context is insufficient, state that the answer cannot be found in the provided text.\n\nContext:\n---\n{context}\n---\n\nQuestion:\n{question}\n\nAnswer:"
            answer = self._generate_response_with_retry(self.config["MODEL_NAME"], answer_prompt)

            if not answer:
                print(f"[{os.getpid()}] Failed to generate answer for this question. Skipping.")
                continue

            print(f"  [{os.getpid()}] [Generated Answer]: {answer[:150]}...")

            qa_pairs.append({"question": question, "answer": answer, "source_chunk_index": i})
            
            if (i + 1) % 10 == 0:
                print(f"\n[{os.getpid()}] Saving intermediate progress with {len(qa_pairs)} pairs to {output_path}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(qa_pairs, f, indent=4)
        
        print(f"\n[{os.getpid()}] Finished processing {doc_name}. Total Q&A pairs generated: {len(qa_pairs)}.")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=4)
        print(f"[{os.getpid()}] Final results saved to {output_path}")

def worker_process(task_args):
    """
    A function executed by each worker process.
    It sets the assigned GPU and processes a single PDF file.
    """
    pdf_path, gpu_id = task_args
    
    # Critical step: Set environment variable to isolate the GPU for this process.
    # This tells CUDA-aware applications which GPU to use.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"--- [Worker PID: {os.getpid()}, Assigned GPU: {gpu_id}] --- Starting job for: {os.path.basename(pdf_path)}")
    
    try:
        # Initialize the generator *within* the worker process
        generator = QAGenerator(CONFIG)
        generator.process_document(pdf_path)
    except Exception as e:
        print(f"--- [Worker PID: {os.getpid()}] --- FATAL ERROR processing {os.path.basename(pdf_path)}: {e}")

def main():
    """Entry point of the script for parallel processing."""
    pdf_files = [f for f in os.listdir(CONFIG["FILES_DIR"]) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in the '{CONFIG['FILES_DIR']}' directory.")
        return

    print(f"Found {len(pdf_files)} PDF files to process: {', '.join(pdf_files)}")
    print(f"Starting parallel processing across {CONFIG['NUM_GPUS']} GPUs.")

    # Create a list of tasks (pdf_path, gpu_id) for the worker pool
    tasks = []
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(CONFIG["FILES_DIR"], pdf_file)
        gpu_id = i % CONFIG['NUM_GPUS'] # Round-robin GPU assignment
        tasks.append((pdf_path, gpu_id))

    # Use a multiprocessing pool to run tasks in parallel
    with multiprocessing.Pool(processes=CONFIG['NUM_GPUS']) as pool:
        pool.map(worker_process, tasks)
    
    print("\nAll documents have been processed.")

if __name__ == "__main__":
    # Ensure multiprocessing works correctly across platforms
    multiprocessing.set_start_method('spawn', force=True)
    main()

