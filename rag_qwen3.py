import collections
import os
import torch
import time
print(f"⏱️  Imports started at {time.time():.2f}")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
print(f"⏱️  Transformers imported at {time.time():.2f}")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from squad_utils import get_squad_item
import json
from utils import parse_needles_jsonl_grouped, extract_text_and_qa_from_needles
print(f"⏱️  All imports completed at {time.time():.2f}")

# Setup cache directories to use existing team cache
def setup_cache_directories():
    cache_dir = "/root/.cache/huggingface"
    os.environ["HF_HOME"] = "/root/.cache"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    print(f"Using existing Hugging Face cache directory: {cache_dir}")

# Cache setup will be called in main() when needed


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
        
        # Initialize the model once in __init__
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize Qwen3-8B model once. This is the heavy operation that should only happen once.
        """
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
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
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading for insufficient GPU RAM
        )
        
        # Load model with exact same config as training script
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")

    def _split_text_into_chunks(self, text: str, split_mode: str = "both", min_length: int = 20):
        """
        Split text into chunks using different strategies.
        
        Args:
            text (str): The input text to split
            split_mode (str): How to split the text
                - "paragraph": Split only by newlines (\n)
                - "sentence": Split only by periods (.)
                - "both": Split by newlines then by periods (default)
            min_length (int): Minimum character length for chunks (default: 5)
                              The original 20 was too restrictive - many valid short answers were lost
                
        Returns:
            list: List of text chunks
        """
        chunks = []
        
        if split_mode == "paragraph":
            # Split only by newlines
            for paragraph in text.split('\n'):
                paragraph = paragraph.strip()
                if paragraph and len(paragraph) >= min_length:
                    chunks.append(paragraph)
                    
        elif split_mode == "sentence":
            # Split only by periods
            sentences = text.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) >= min_length:
                    chunks.append(sentence)
                    
        elif split_mode == "both":
            # Split by newlines, then by periods (original behavior)
            for paragraph in text.split('\n'):
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) >= min_length:
                        chunks.append(sentence)
        else:
            raise ValueError(f"Invalid split_mode: {split_mode}. Use 'paragraph', 'sentence', or 'both'")
        
        return chunks

    def setup_from_text(self, text: str, split_mode: str = "both", min_length: int = 20):
        """
        Process the given text for retrieval. Model is already loaded in __init__.
        
        Args:
            text (str): The input text to process
            split_mode (str): How to split text - "paragraph", "sentence", or "both"
            min_length (int): Minimum character length for chunks (default: 5 instead of 20)
        """
        # Split text into documents using the configurable method
        print(f"📄 Processing documents (split_mode: {split_mode}, min_length: {min_length})...")
        self.documents = self._split_text_into_chunks(text, split_mode, min_length)
        
        print(f"Created {len(self.documents)} document chunks")
        
        # Create TF-IDF vectors for retrieval (simpler than sentence transformers)
        print("🔍 Creating TF-IDF vectors for retrieval...")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        print("✅ TF-IDF vectors created")
        print("📄 Document processing complete!")
        
        # Show GPU memory usage after setup
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
    # Setup cache directories when actually needed
    setup_cache_directories()
    
    # Initialize model ONCE - this is the heavy operation
    print("🚀 Initializing RAG model (this happens only once)...")
    rag_model = RAGModelQwen3()
    res = {}
    
    for split_mode in ["paragraph", "sentence", "both"]:
        for path in ["needles/2048/qa_1_2048.jsonl","needles/32768/qa_1_32768.jsonl","needles/131072/qa_1_131072.jsonl"]:  # Start with 2048 first
            count_correct = 0
            docs_data = extract_text_and_qa_from_needles(path)
            total_questions = 0
            for doc in docs_data[:70]:  # Process first 10 items for demo
                # Only process documents - model is already loaded
                rag_model.setup_from_text(doc["document"], split_mode=split_mode)

                q = doc["question"]
                a = doc["answers"]

                answer = rag_model.ask(q)
                print(f"Q: {q}")
                print(f"A: {answer}")
                print(f"Expected: {a}")
                print("---")
                
                # Check if any expected answer appears in the response
                count_correct += 1 if any(ans.lower() in answer.lower() for ans in a) else 0
                total_questions += 1

            accuracy = count_correct / total_questions if total_questions > 0 else -1
            print(f"Results for file at path {path}:")
            print(f"Accuracy on {path}: {count_correct}/{total_questions} = {accuracy:.2%}")
            res[(path, split_mode)] = accuracy
        
    print("Final aggregated results:")
    for key, value in res.items():
        print(f"{key}: {value:.2%}")


def main_2048():
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