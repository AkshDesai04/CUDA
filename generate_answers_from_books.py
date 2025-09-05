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
    "QUESTIONS_DIR": "./questions_from_ebooks",
    "OUTPUT_ANSWERS_DIR": "./answers_from_ebooks",
    "FAISS_K": 4  # Number of relevant chunks to retrieve
}

class QAFromPDF:
    def __init__(self, config):
        self.config = config
        self.client = ollama.Client()
        self._check_models()
        os.makedirs(self.config["OUTPUT_ANSWERS_DIR"], exist_ok=True)

    def _check_models(self):
        """Ensure the required models are pulled."""
        try:
            models_list = [m.get('name') for m in self.client.list().get('models', [])]
            if f"{self.config['MODEL_NAME']}:latest" not in models_list:
                print(f"Pulling {self.config['MODEL_NAME']}...")
                self.client.pull(self.config['MODEL_NAME'])
            if f"{self.config['EMBED_MODEL_NAME']}:latest" not in models_list:
                print(f"Pulling {self.config['EMBED_MODEL_NAME']}...")
                self.client.pull(self.config['EMBED_MODEL_NAME'])
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            sys.exit(1)

    def _load_chunks_and_index(self, pdf_name):
        """Load FAISS index and text chunks for a PDF."""
        base = os.path.splitext(pdf_name)[0]
        index_path = os.path.join(self.config["VDB_DIR"], f"{base}.faiss")
        chunks_path = os.path.join(self.config["VDB_DIR"], f"{base}.chunks.json")
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            print(f"VDB missing for {pdf_name}. Skipping.")
            return None, None

        index = faiss.read_index(index_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return index, chunks

    def _get_embedding(self, text):
        """Get embedding vector for a given text."""
        try:
            return np.array(self.client.embeddings(model=self.config["EMBED_MODEL_NAME"], prompt=text)["embedding"], dtype='float32')
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(1024, dtype='float32')

    def _retrieve_relevant_chunks(self, question, index, chunks, k):
        """Retrieve top-k relevant chunks using FAISS."""
        q_emb = self._get_embedding(question).reshape(1, -1)
        D, I = index.search(q_emb, k)
        return [chunks[i] for i in I[0] if i < len(chunks)]

    def _generate_answer(self, question, context_chunks):
        """Generate answer constrained to the context chunks."""
        prompt = f"Answer the following question using ONLY the text below from the PDF. Do not use external knowledge.\n\nContext:\n---\n{''.join(context_chunks)}\n---\nQuestion: {question}\nAnswer:"
        try:
            response = self.client.generate(model=self.config["MODEL_NAME"], prompt=prompt, stream=False)
            return response['response'].strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""

    def process_all(self):
        """Process all PDFs and question JSONs."""
        question_files = [f for f in os.listdir(self.config["QUESTIONS_DIR"]) if f.endswith(".json")]

        for q_file in question_files:
            print(f"\nProcessing questions from {q_file}...")
            base_name = os.path.splitext(q_file)[0]
            pdf_path = os.path.join(self.config["FILES_DIR"], f"{base_name}.pdf")

            if not os.path.exists(pdf_path):
                print(f"PDF not found for {q_file}. Skipping.")
                continue

            index, chunks = self._load_chunks_and_index(f"{base_name}.pdf")
            if index is None:
                continue

            with open(os.path.join(self.config["QUESTIONS_DIR"], q_file), 'r', encoding='utf-8') as f:
                questions = json.load(f)

            answers = []
            for q_obj in questions:
                question = q_obj['question']
                relevant_chunks = self._retrieve_relevant_chunks(question, index, chunks, self.config["FAISS_K"])
                answer = self._generate_answer(question, relevant_chunks)
                answers.append({"question": question, "answer": answer})

            output_path = os.path.join(self.config["OUTPUT_ANSWERS_DIR"], f"{base_name}.answers.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(answers, f, indent=4)
            print(f"Answers saved to {output_path}")

def main():
    qa_system = QAFromPDF(CONFIG)
    qa_system.process_all()

if __name__ == "__main__":
    main()
