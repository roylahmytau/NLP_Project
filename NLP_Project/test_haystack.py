# Test script to verify Haystack installation
import sys
print(f"Python version: {sys.version}")

try:
    from haystack.document_stores import InMemoryDocumentStore
    print("✓ InMemoryDocumentStore imported successfully")
except ImportError as e:
    print(f"✗ Failed to import InMemoryDocumentStore: {e}")

try:
    from haystack.nodes import EmbeddingRetriever, FARMReader
    print("✓ EmbeddingRetriever and FARMReader imported successfully")
except ImportError as e:
    print(f"✗ Failed to import nodes: {e}")

try:
    from haystack.pipelines import ExtractiveQAPipeline
    print("✓ ExtractiveQAPipeline imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pipeline: {e}")

try:
    from haystack.utils import convert_files_to_docs
    print("✓ convert_files_to_docs imported successfully")
except ImportError as e:
    print(f"✗ Failed to import utils: {e}")

try:
    import sentence_transformers
    print(f"✓ sentence-transformers version: {sentence_transformers.__version__}")
except ImportError as e:
    print(f"✗ Failed to import sentence-transformers: {e}")

try:
    import transformers
    print(f"✓ transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Failed to import transformers: {e}")

print("\nIf all imports show ✓, your environment is ready!")