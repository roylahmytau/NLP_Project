import os
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, PromptNode, PromptTemplate
from haystack.pipelines import Pipeline
from haystack import Document
from haystack.nodes.prompt import PromptModel
from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer


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
        self.document_store = InMemoryDocumentStore(use_bm25=True)
        # Split text into paragraphs and create documents
        paragraphs = text.split('\n\n')
        documents = [Document(content=p) for p in paragraphs if p.strip()]
        self.document_store.write_documents(documents)

        retriever = BM25Retriever(document_store=self.document_store)

        # The PromptNode will download the model from Hugging Face if not cached.
        prompt_template = PromptTemplate(
            prompt='''Given the provided Documents, answer the Question.
                       If the answer is not in the Documents, say 'I don't know'.
                       Documents: {join(documents)}
                       Question: {query}
                       Answer:''',
            output_parser=None
        )

        prompt_model = PromptModel(
            model_name_or_path="google/flan-t5-base",
            invocation_layer_class=HFLocalInvocationLayer,
            model_kwargs={"device": "cpu", "task_name": "text2text-generation"} # Assumes CPU, change to "cuda" for GPU
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

        result = self.pipeline.run(query=question, params={"Retriever": {"top_k": 3}})

        # The output of a PromptNode is a list of strings in the 'results' key.
        if result and result.get('results'):
            return result['results'][0]
        return "No answer found."


if __name__ == '__main__':
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
    file_path = r'C:\Users\ohads\Desktop\NLP\final project\code\NLP_Project\books\short_book.txt'
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