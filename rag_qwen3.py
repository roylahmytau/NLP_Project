import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from squad_utils import get_squad_item

# Setup cache directories to use existing team cache
def setup_cache_directories():
    cache_dir = "/root/.cache/huggingface"
    os.environ["HF_HOME"] = "/root/.cache"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    print(f"Using existing Hugging Face cache directory: {cache_dir}")

# Setup cache immediately
setup_cache_directories()


class RAGModelQwen3:
    """
    A RAG-based model using Qwen3-8B with the exact same loading approach as train_lora_optimized.py
    """

    def __init__(self):
        self.documents = []
        self.vectorizer = None
        self.doc_vectors = None
        self.tokenizer = None
        self.model = None
        self.device = None

    def setup_from_text(self, text: str):
        """
        Sets up the RAG pipeline with the given text using Qwen3-8B exactly like train_lora_optimized.py
        """
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Split text into documents
        print("📄 Processing documents...")
        self.documents = []
        for p in text.split('\n'):
            # Split by sentences
            sentences = p.split('. ')
            for sentence in sentences:
                if sentence.strip() and len(sentence.strip()) > 20:
                    self.documents.append(sentence.strip())
        
        print(f"Created {len(self.documents)} document chunks")
        
        # Create TF-IDF vectors for retrieval (simpler than sentence transformers)
        print("🔍 Creating TF-IDF vectors for retrieval...")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        print("✅ TF-IDF vectors created")

        # Load Qwen3-8B model EXACTLY like train_lora_optimized.py
        print("📥 Loading Qwen3-8B model using train_lora_optimized.py approach...")
        
        # Load tokenizer with trust_remote_code like training script
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use 4-bit quantization like training script for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model with exact same config as training script
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Qwen3-8B model loaded successfully!")
        print("RAG pipeline is set up and ready.")
        
        # Show GPU memory usage
        if torch.cuda.is_available():
            print(f"GPU Memory after setup: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, {torch.cuda.memory_reserved()/1024**2:.1f} MB reserved")

    def retrieve_documents(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve the most relevant documents for a query using TF-IDF and optimized top-k selection.
        """
        if self.doc_vectors is None:
            raise Exception("Model not set up. Call setup_from_text first.")
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities - this gives us a scipy sparse matrix
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top-k most similar documents efficiently - O(n + k log k) instead of O(n log n)
        top_indices = heapq.nlargest(top_k, range(len(similarities)), key=lambda i: similarities[i])
        
        retrieved_docs = []
        for idx in top_indices:
            retrieved_docs.append({
                'text': self.documents[idx],
                'score': similarities[idx]
            })
        
        return retrieved_docs

    def ask(self, question: str) -> str:
        """
        Asks a question to the RAG model using Qwen3-8B.
        """
        if not self.model or not self.tokenizer:
            raise Exception("Pipeline not set up. Please call 'setup_from_text' first.")

        # Monitor GPU usage before inference
        if torch.cuda.is_available():
            print(f"GPU Memory before inference: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(question, top_k=3)
        context = "\n".join([doc['text'] for doc in retrieved_docs])
        
        # Create prompt in Qwen instruction format (like train_lora_optimized.py)
        prompt = f"<|im_start|>user\nGiven the provided context, answer the question.\nIf the answer is not in the context, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}\n<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer (remove the prompt)
        answer = response[len(prompt):].strip()
        
        # Monitor GPU usage after inference
        if torch.cuda.is_available():
            print(f"GPU Memory after inference: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

        return answer


def main():
    # Fetch the document from squad_utils
    script_dir = os.path.dirname(os.path.abspath(__file__))
    squad_path = os.path.join(script_dir, "squad.json")
    docs = ""
    answers = []
    questions = []
    for i in range(1):
        docs += f"doc{i} " + get_squad_item(squad_path, i, "document") + "\n\n"
        answers.extend(get_squad_item(squad_path, i, "answers"))
        questions.extend(get_squad_item(squad_path, i, "questions"))

    rag_model = RAGModelQwen3()
    rag_model.setup_from_text(docs)
    
    count_correct = 0
    total_questions = len(questions)
    for q, a in zip(questions, answers):
        res = rag_model.ask(q)
        print(f"Q: {q}")
        print(f"A: {res}")
        print(f"Expected: {a}")
        print("---")
        # Check if any expected answer appears in the response
        count_correct += 1 if any(ans.lower() in res.lower() for ans in a) else 0

    print(f"Accuracy on SQuAD doc 0: {count_correct}/{total_questions} = {count_correct/total_questions:.2%}")


if __name__ == '__main__':
    main()