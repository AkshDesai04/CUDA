import os
import json
import ollama
import faiss
import numpy as np
import PyPDF2
import time
import sys

# --- Configuration ---
CONFIG = {
    "MODEL_NAME": "gemma3:27b",
    "EMBED_MODEL_NAME": "mxbai-embed-large",
    "FILES_DIR": "./files",
    "VDB_DIR": "./vdb",
    "OUTPUT_JSON_DIR": "./output_json",
    "CHUNK_SIZE": 500,  # words
    "CHUNK_OVERLAP": 1000, # words
    "FAISS_K": 4, # Number of relevant chunks to retrieve for answering
}

class QAGenerator:
    """
    A class to generate Question-Answer pairs from PDF documents using a RAG pipeline.
    It separates the process into two phases: VDB creation and Q&A generation.
    """
    def __init__(self, config):
        self.config = config
        self.client = ollama.Client()
        self._check_models()
        os.makedirs(self.config["VDB_DIR"], exist_ok=True)
        os.makedirs(self.config["OUTPUT_JSON_DIR"], exist_ok=True)
        print("QAGenerator initialized.")

    def _check_models(self):
        """Ensures the required models are available in Ollama."""
        try:
            ollama_response = self.client.list()
            models_list = ollama_response.get('models', [])
            models_available = [m.get('name') for m in models_list if m and m.get('name')]

            if f"{self.config['MODEL_NAME']}:latest" not in models_available:
                print(f"Model '{self.config['MODEL_NAME']}' not found. Pulling it now...")
                self.client.pull(self.config['MODEL_NAME'])
            if f"{self.config['EMBED_MODEL_NAME']}:latest" not in models_available:
                print(f"Embedding model '{self.config['EMBED_MODEL_NAME']}' not found. Pulling it now...")
                self.client.pull(self.config['EMBED_MODEL_NAME'])
        except Exception as e:
            print(f"Error connecting to Ollama or pulling models: {e}")
            print("Please ensure Ollama is running and accessible.")
            sys.exit(1)

    def _load_and_chunk_pdf(self, pdf_path):
        """Loads text from a PDF and splits it into overlapping chunks."""
        print(f"  - Loading and chunking '{os.path.basename(pdf_path)}'...")
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.config["CHUNK_SIZE"] - self.config["CHUNK_OVERLAP"]):
                chunk = " ".join(words[i:i + self.config["CHUNK_SIZE"]])
                chunks.append(chunk)
            print(f"  - Document split into {len(chunks)} chunks with overlap.")
            return chunks
        except Exception as e:
            print(f"  - Error reading or chunking PDF {pdf_path}: {e}")
            return []

    def _get_embeddings(self, texts, doc_name):
        """Generates embeddings for a list of text chunks."""
        print(f"  - Generating embeddings for {len(texts)} chunks of '{doc_name}'...")
        embeddings = []
        for i, text in enumerate(texts):
            try:
                response = self.client.embeddings(model=self.config["EMBED_MODEL_NAME"], prompt=text)
                embeddings.append(response["embedding"])
                print(f"    - Embedded chunk {i+1}/{len(texts)}", end='\r')
            except Exception as e:
                print(f"\n    - Error generating embedding for chunk {i+1}: {e}")
                embeddings.append([0.0] * 1024) # mxbai-embed-large has 1024 dimensions
        print(f"\n  - Embedding generation complete for '{doc_name}'.")
        return np.array(embeddings).astype('float32')

    def build_and_save_vdb(self, pdf_path):
        """Creates and saves the Vector Database (FAISS index) and chunks for a PDF."""
        doc_name = os.path.basename(pdf_path)
        base_filename = os.path.splitext(doc_name)[0]
        
        index_path = os.path.join(self.config["VDB_DIR"], f"{base_filename}.faiss")
        chunks_path = os.path.join(self.config["VDB_DIR"], f"{base_filename}.chunks.json")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            print(f"VDB for '{doc_name}' already exists. Skipping build.")
            return

        chunks = self._load_and_chunk_pdf(pdf_path)
        if not chunks:
            return

        embeddings = self._get_embeddings(chunks, doc_name)
        if embeddings.shape[0] == 0:
            print(f"  - No embeddings generated for '{doc_name}'. Cannot create VDB.")
            return

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        print(f"  - Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)
        
        print(f"  - Saving text chunks to {chunks_path}")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4)
            
        print(f"VDB for '{doc_name}' created successfully.")

    def _generate_response_with_retry(self, model, prompt, max_retries=3):
        """Generates a response from the LLM with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.generate(model=model, prompt=prompt, stream=False)
                return response['response'].strip()
            except Exception as e:
                print(f"\nLLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5 * (attempt + 1))
        return None

    def generate_qa_from_vdb(self, pdf_file):
        """Generates CUDA/coding-related questions from a pre-built VDB."""
        base_filename = os.path.splitext(pdf_file)[0]
        index_path = os.path.join(self.config["VDB_DIR"], f"{base_filename}.faiss")
        chunks_path = os.path.join(self.config["VDB_DIR"], f"{base_filename}.chunks.json")
        output_path = os.path.join(self.config["OUTPUT_JSON_DIR"], f"{base_filename}.json")

        print(f"\n--- Generating CUDA/Coding Questions for {pdf_file} ---")

        try:
            print(f"  - Loading FAISS index from {index_path}")
            index = faiss.read_index(index_path)
            print(f"  - Loading text chunks from {chunks_path}")
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"Error loading VDB files for '{pdf_file}': {e}. Skipping.")
            return

        questions = []
        total_chunks = len(chunks)
        print(f"  - Starting question generation for {total_chunks} source chunks...")

        for i, chunk in enumerate(chunks):
            print(f"\n  --- Processing Chunk {i+1}/{total_chunks} for {pdf_file} ---")
            
            question_prompt = f"Based ONLY on the following text, generate one clear, detailed question specifically related to CUDA, coding, or related technical topics. Do not answer it. Output only the question.\n\nText:\n---\n{chunk}\n---\n\nQuestion:"
            question = self._generate_response_with_retry(self.config["MODEL_NAME"], question_prompt)

            if not question:
                print(f"  - Failed to generate question for this chunk. Skipping.")
                continue
            
            print(f"    [Generated Question]: {question}")
            questions.append({"question": question, "source_chunk_index": i})
            
            if (i + 1) % 10 == 0:
                print(f"\n  - Saving intermediate progress with {len(questions)} questions to {output_path}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(questions, f, indent=4)
        
        print(f"\nFinished processing {pdf_file}. Total questions generated: {len(questions)}.")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=4)
        print(f"Final results saved to {output_path}")

def main():
    """Entry point of the script."""
    generator = QAGenerator(CONFIG)
    
    pdf_files = [f for f in os.listdir(CONFIG["FILES_DIR"]) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in the '{CONFIG['FILES_DIR']}' directory.")
        return

    print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")

    # Phase 1: Create all Vector Databases first
    print(f"\n{'='*25} Phase 1: Building Vector Databases {'='*25}")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(CONFIG["FILES_DIR"], pdf_file)
        generator.build_and_save_vdb(pdf_path)
    
    # Phase 2: Generate Q&A pairs from the created VDBs
    print(f"\n{'='*25} Phase 2: Generating CUDA/Coding Questions {'='*25}")
    for pdf_file in pdf_files:
        generator.generate_qa_from_vdb(pdf_file)

    print("\nAll documents have been processed.")

if __name__ == "__main__":
    main()
