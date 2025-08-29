#!/usr/bin/env python3
import json, re, argparse
from typing import Tuple

SPLIT_INSTR_REGEX = re.compile(r"^(.*?\bdo not output any other words\.)", flags=re.IGNORECASE | re.DOTALL)
SPLIT_QUESTION_START_REGEX = re.compile(r"\bAnswer the question based on the given documents\b", flags=re.IGNORECASE | re.DOTALL)

def split_input(text: str) -> Tuple[str, str, str]:
    m_instr = SPLIT_INSTR_REGEX.search(text)
    if m_instr:
        instruction = m_instr.group(1)
        rest = text[m_instr.end():]
    else:
        instruction, rest = "", text

    m_q = SPLIT_QUESTION_START_REGEX.search(rest)
    if m_q:
        doc = rest[:m_q.start()]
        question = rest[m_q.start():]
    else:
        doc, question = rest, ""

    instruction = instruction.rstrip("\n")
    doc = doc.strip("\n")
    question = question.lstrip("\n")
    return instruction, doc, question

def main():

    read_count = 0
    write_count = 0
    for i in [2048, 32768, 131072]:
        with open("RULER/runs/my-hf/synthetic/" + str(i) + "/data/qa_1/validation.jsonl", "r", encoding="utf-8") as fin, open('qa_1_'+ str(i) + '.jsonl', "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                read_count += 1
                obj = json.loads(line)
                text = obj.pop("input", "")
                instruction, doc, question = split_input(text)
                obj["instruction"] = instruction
                obj["doc"] = doc
                obj["question"] = question
                fout.write(json.dumps(obj, ensure_ascii=False) + "\\n")
                write_count += 1

        print(f"Done. Read {read_count} lines, wrote {write_count} lines")

if __name__ == "__main__":
    main()
