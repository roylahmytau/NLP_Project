"""
Utility helpers to inspect SQuAD v2.0 files.

Provides `get_squad_item` to fetch, by paragraph index, either the document
context, the list of questions, or the list of lists of potential answers.
"""

from typing import List, Literal, Union
import json
from statistics import mean


Kind = Literal["document", "questions", "answers"]


def get_squad_item(file_path: str, index: int, kind: Kind) -> Union[str, List[str], List[List[str]]]:
    """Return the requested SQuAD item for the given paragraph index.

    Args:
        file_path: Path to SQuAD v2.0 JSON file (e.g., squad.json).
        index: Zero-based paragraph index across all articles/paragraphs.
        kind: One of "document", "questions", or "answers".

    Returns:
        - If kind == "document": the paragraph context string
        - If kind == "questions": list of question strings in that paragraph (excluding is_impossible=True)
        - If kind == "answers": list (per question) of lists of answer strings (excluding is_impossible=True)

    Raises:
        ValueError: If kind is invalid or index is out of range.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    paragraphs = []
    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            paragraphs.append(paragraph)

    if index < 0 or index >= len(paragraphs):
        raise ValueError(f"index {index} out of range for {len(paragraphs)} paragraphs")

    paragraph = paragraphs[index]

    if kind == "document":
        return paragraph.get("context", "")
    elif kind == "questions":
        return [
            qa.get("question", "")
            for qa in paragraph.get("qas", [])
            if not qa.get("is_impossible", False)
        ]
    elif kind == "answers":
        return [
            [ans.get("text", "") for ans in qa.get("answers", [])]
            for qa in paragraph.get("qas", [])
            if not qa.get("is_impossible", False)
        ]
    else:
        raise ValueError("kind must be one of 'document', 'questions', or 'answers'")


if __name__ == "__main__":
    # Small CLI for convenience
    import argparse
    parser = argparse.ArgumentParser(description="Inspect SQuAD by paragraph index")
    parser.add_argument("file", type=str, help="Path to squad.json")
    parser.add_argument("index", type=int, help="Zero-based paragraph index")
    parser.add_argument("kind", choices=["document", "questions", "answers"], help="Item to return")
    args = parser.parse_args()

    result = get_squad_item(args.file, args.index, args.kind)  # type: ignore[arg-type]
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


