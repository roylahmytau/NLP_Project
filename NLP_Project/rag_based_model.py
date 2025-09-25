import os
import torch
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, PromptNode, PromptTemplate
from haystack.pipelines import Pipeline
from haystack import Document
from haystack.nodes.prompt import PromptModel
from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer
from squad_utils import get_squad_item

# Setup cache directories to avoid disk quota issues
def setup_cache_directories():
    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "..", "..", ".huggingface_cache")
    cache_dir = os.path.abspath(cache_dir)  # Convert to absolute path
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables for all caching
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    print(f"Using Hugging Face cache directory: {cache_dir}")

# Setup cache immediately
setup_cache_directories()
import torch
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, PromptNode, PromptTemplate
from haystack.pipelines import Pipeline
from haystack import Document
from haystack.nodes.prompt import PromptModel
from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer
from squad_utils import get_squad_item


class RAGModel:
    """
    A RAG-based model for question answering on a given text.
    """

    def __init__(self):
        self.pipeline = None
        self.document_store = None

    def setup_from_text(self, text: str):
        """
        Sets up the RAG pipeline with the given text.

        :param text: A string containing the text to be used as a knowledge base.
        """
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.document_store = InMemoryDocumentStore(use_bm25=True)
        # Split text into paragraphs and create documents with  \n or . or , as delimiters
        paragraphs = []
        for p in text.split('\n'):
            paragraphs.extend(p.split('. '))
            paragraphs.extend(p.split(', '))
        documents = [Document(content=p) for p in paragraphs if p.strip()]
        self.document_store.write_documents(documents)

        retriever = BM25Retriever(document_store=self.document_store)

        # The PromptNode will download the model from Hugging Face if not cached.
        prompt_template = PromptTemplate(
            prompt='''Given the provided Documents, answer the Question.
                       If the answer is not in the Documents, say 'I don\'t know'.
                       Documents: {join(documents)}
                       Question: {query}
                       Answer:''',
            output_parser=None
        )

        # Check GPU compatibility and choose device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name}")
            # Note: Even if TITAN Xp shows warnings, we'll try to use it anyway
            print(f"Will attempt to use GPU despite any compatibility warnings.")
        
        print(f"Using device: {device}")
        
        # Use Qwen model - fix the model name and task type
        prompt_model = PromptModel(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            invocation_layer_class=HFLocalInvocationLayer,
            model_kwargs={"device": device, "task_name": "text-generation"}  # Changed to text-generation for Qwen
        )

        prompt_node = PromptNode(
            model_name_or_path=prompt_model,
            default_prompt_template=prompt_template,
            max_length=200,
        )

        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
        print("RAG pipeline is set up and ready.")
        
        # Show current GPU usage if available
        if torch.cuda.is_available():
            print(f"GPU Memory after setup: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, {torch.cuda.memory_reserved()/1024**2:.1f} MB reserved")

    def setup_from_file(self, file_path: str):
        """
        Sets up the RAG pipeline with text from a file.

        :param file_path: The absolute path to the .txt file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file type. Please use .txt files only.")

        self.setup_from_text(text)

    def ask(self, question: str) -> str:
        """
        Asks a question to the RAG model.

        :param question: The question to ask.
        :return: The answer from the model.
        """
        if not self.pipeline:
            raise Exception("Pipeline not set up. Please call 'setup_from_text' or 'setup_from_file' first.")

        # Monitor GPU usage before inference
        if torch.cuda.is_available():
            print(f"GPU Memory before inference: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        
        result = self.pipeline.run(query=question, params={"Retriever": {"top_k": 1}})
        
        # Monitor GPU usage after inference
        if torch.cuda.is_available():
            print(f"GPU Memory after inference: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

        # The output of a PromptNode is a list of strings in the 'results' key.
        if result and result.get('results'):
            return result['results'][0]
        return "No answer found."


def main():
    # fetch the document of doc 0 from squad_utils
    script_dir = os.path.dirname(os.path.abspath(__file__))
    squad_path = os.path.join(script_dir, "books", "squad.json")
    docs = ""
    answers = []
    questions = []
    for i in range(1):
        docs += f"doc{i} " +get_squad_item(squad_path, i, "document") + "\n\n"
        answers += get_squad_item(squad_path, i, "answers")
        questions += get_squad_item(squad_path, i, "questions")


    rag_model_string = RAGModel()
    rag_model_string.setup_from_text(docs)
    count_correct = 0
    total_questions = len(questions)
    for q, a in zip(questions, answers):
        res = rag_model_string.ask(q)
        count_correct += 1 if res in a else 0

    print(f"Accuracy on SQuAD doc 0: {count_correct}/{total_questions} = {count_correct/total_questions:.2%}")


    


def main1():
    # --- Example with a string ---
    print("--- Running example with a string ---")
    rag_model_string = RAGModel()

    sample_text = '''
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially
    criticized by some of France's leading artists and intellectuals for its design, but it
    has become a global cultural icon of France and one of the most recognizable structures in the world.
    The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building,
    and is the tallest structure in Paris.
    '''

    rag_model_string.setup_from_text(sample_text)

    q1 = "Who designed the Eiffel Tower?"
    ans1 = rag_model_string.ask(q1)
    print(f"Q: {q1}")
    print(f"A: {ans1}")

    # --- Example with a file ---
    # Using the .txt file as requested.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "books", "short_book.txt")
    if os.path.exists(file_path):
        print(f"\n--- Running example with file: {os.path.basename(file_path)} ---")
        rag_model_file = RAGModel()
        rag_model_file.setup_from_file(file_path)

        q2 = "What is the Eiffel Tower made of?"
        ans2 = rag_model_file.ask(q2)
        print(f"Q: {q2}")
        print(f"A: {ans2}")

    else:
        print(f"\n--- File example skipped ---")
        print(f"File not found: {file_path}")

def main_long_book():
    """
    Reads from 'book text.txt' and asks questions about it.
    """
    print("\n--- Running example with file: book text.txt ---")
    rag_model_book = RAGModel()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "books", "book text.txt")

    try:
        rag_model_book.setup_from_file(file_path)

        questions = [
            "Who is the author of the book?",
            "What is the synopsis of Dangerous Liaisons?",
            "Who does Cécile de Volanges fall in love with?",
            "What is Viscount de Valmont's main goal?",
            "What is the relationship between the Marchioness de Merteuil and Gercourt?"
        ]

        for q in questions:
            print(f"\nQ: {q}")
            ans = rag_model_book.ask(q)
            print(f"A: {ans}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    #main1()
    main()
