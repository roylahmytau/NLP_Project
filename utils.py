#from train_lora_optimized import load_qa_data
import json


def parse_needles_qa_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Load the entire JSON array
    
    print(f"Successfully parsed {len(data)} documents from {file_path}")
    

    
    return data

def parse_needles_jsonl_grouped(file_path):
    """
    Parse needles JSONL file and group by individual documents.
    Groups questions that have the same document content using MD5 hash as key.
    
    Args:
        file_path (str): Path to the needles JSONL file
        
    Returns:
        list: List of dictionaries, each containing:
            - document: the individual document text
            - questions: list of questions for this document  
            - answers: list of lists of answers (each question can have multiple expected answers)
            - instruction: instruction for answering
            - original_indices: list of original indices from JSONL
    """
    import hashlib
    
    # First, load all JSONL entries
    jsonl_entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # Check if it's a single JSON array
        if content.startswith('['):
            try:
                jsonl_entries = json.loads(content)
                print(f"Loaded as JSON array")
            except json.JSONDecodeError as e:
                print(f"Error parsing as JSON array: {e}")
                return []
        else:
            # Parse comma-separated JSON objects one by one
            decoder = json.JSONDecoder()
            pos = 0
            
            while pos < len(content):
                # Skip whitespace and commas
                while pos < len(content) and content[pos] in ' \t\n\r,':
                    pos += 1
                
                if pos >= len(content):
                    break
                    
                try:
                    obj, end_pos = decoder.raw_decode(content, pos)
                    jsonl_entries.append(obj)
                    pos += end_pos  # Move to the position after this object
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse object at position {pos}: {e}")
                    # Try to skip to next potential object start
                    next_brace = content.find('{', pos + 1)
                    if next_brace == -1:
                        break
                    pos = next_brace
            
            print(f"Parsed {len(jsonl_entries)} comma-separated JSON objects")
    
    print(f"Loaded {len(jsonl_entries)} entries from JSONL file")
    
    # Group documents using MD5 hash as key
    document_groups = {}

    for entry in jsonl_entries:
        doc_text = entry['doc']
        
        # Clean up the document text - remove intro and document labels
        import re
        # Remove "The following are given documents." and similar intro text
        cleaned_doc = re.sub(r'^The following are given documents\.\s*\n\n', '', doc_text)
        # Remove "Document X:" labels but keep the content
        cleaned_doc = re.sub(r'Document \d+:\s*\n?', '', cleaned_doc)
        # Clean up extra whitespace
        cleaned_doc = re.sub(r'\n\s*\n', '\n\n', cleaned_doc).strip()
        
        # Create MD5 hash of cleaned document text as key
        doc_hash = hashlib.md5(cleaned_doc.encode()).hexdigest()
                       
        # Use document hash as grouping key
        if doc_hash not in document_groups:
            document_groups[doc_hash] = {
                'document': cleaned_doc,
                'questions': [],
                'answers': [],  # List of lists - each question can have multiple answers
                'instruction': entry.get('instruction', '')
            }
        
        # Add this question and its answers to the group
        document_groups[doc_hash]['questions'].append(entry['question'])
        document_groups[doc_hash]['answers'].append(entry['outputs'])  # outputs is already a list

    # Convert to list

    return list(document_groups.values())

def extract_text_and_qa_from_needles(file_path):
    """
    Extract text content and Q&A pairs without JSON parsing issues.
    Since there's one question per document, return simple list of dicts.
    """
    import re
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract all doc, question, and outputs using regex
    doc_pattern = r'"doc":\s*"([^"]*(?:\\.[^"]*)*)"'
    question_pattern = r'"question":\s*"([^"]*(?:\\.[^"]*)*)"'
    outputs_pattern = r'"outputs":\s*(\[[^\]]*\])'
    
    docs = re.findall(doc_pattern, content)
    questions = re.findall(question_pattern, content)  
    outputs_raw = re.findall(outputs_pattern, content)
    
    # Parse outputs as JSON arrays
    outputs = []
    for out_str in outputs_raw:
        try:
            outputs.append(json.loads(out_str))
        except:
            outputs.append([])  # fallback
    
    # Combine into list of dicts
    qa_pairs = []
    for i in range(min(len(docs), len(questions), len(outputs))):
        # Unescape JSON strings
        doc_text = docs[i].replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        question_text = questions[i].replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        
        qa_pairs.append({
            'document': doc_text,
            'question': question_text,
            'answers': outputs[i]
        })
    
    print(f"Extracted {len(qa_pairs)} complete Q&A pairs from {file_path}")
    return qa_pairs

def main():
    # Test the new JSONL parser that groups by individual documents
    jsonl_path = "needles/32768/qa_1_32768.jsonl"
    grouped_docs = extract_text_and_qa_from_needles(jsonl_path)
    
    # Print overview of grouped documents
    if grouped_docs:
        print(f"\nGrouped Documents Overview:")
        for i, doc_group in enumerate(grouped_docs[:5]):  # Show first 5
            print(f"\nDocument {i+1}:")
            print(f"  - Questions: {len(doc_group['questions'])}")
            print(f"  - Length: {len(doc_group['document'])} chars")
            print(f"  - Sample question: {doc_group['questions'][0][:80]}...")
            print(f"  - Sample answers: {doc_group['answers'][0]}")
            print(f"  - Document preview: {doc_group['document'][:150]}...")
        
        if len(grouped_docs) > 5:
            print(f"\n... and {len(grouped_docs) - 5} more documents")
    
    return grouped_docs



if __name__ == '__main__':
    #main_lora()
    main()