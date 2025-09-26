#!/usr/bin/env python3
"""
Script to convert all JSONL files in needles directory to JSON format in new_needles directory
"""
import json
import os
import argparse
import re
from pathlib import Path

def convert_qa_jsonl(input_file, output_file):
    """Convert QA-style JSONL file (concatenated JSON objects) to JSON array format"""
    objects = []
    
    with open(input_file, 'r') as f:
        content = f.read().strip()
    
    # Split by the pattern where one object ends and another begins
    # Look for } followed by {"index"
    parts = re.split(r'{"index"', content)
    # remove the first empty part
    parts = parts[1:]
    print(f"Split into {len(parts)} parts")
    
    # Process each part
    for i, part in enumerate(parts):
        # Add back the opening brace and index for all parts except the first
        part = '{"index"' + part
        # remove last char for all parts except the last
        if i != len(parts) - 1:
            part = part[:-1]

        try:
            obj = json.loads(part)
            objects.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing part {i+1}: {e}")
            continue
    
    print(f"Successfully parsed {len(objects)} JSON objects from {input_file}")
    return objects

def convert_standard_jsonl(input_file, output_file):
    """Convert standard JSONL file (one JSON object per line) to JSON array format"""
    objects = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    obj = json.loads(line)
                    objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
    
    print(f"Successfully parsed {len(objects)} JSON objects from {input_file}")
    return objects

def detect_jsonl_type(input_file):
    """Detect if the JSONL file is QA-style (concatenated) or standard (one per line)"""
    with open(input_file, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    
    # Check if it's a single line file (all content on one line)
    if not second_line:
        # Single line file - check if it contains multiple JSON objects
        if '{"index"' in first_line and first_line.count('{"index"') > 1:
            return 'qa'
        else:
            return 'standard'
    else:
        # Multi-line file - standard JSONL format
        return 'standard'

def convert_jsonl_to_json(input_file, output_file):
    """Convert JSONL file to JSON array format"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Detect file type
    file_type = detect_jsonl_type(input_file)
    print(f"Detected file type: {file_type}")
    
    if file_type == 'qa':
        objects = convert_qa_jsonl(input_file, output_file)
    else:
        objects = convert_standard_jsonl(input_file, output_file)
    
    # Write as JSON array
    with open(output_file, 'w') as f:
        json.dump(objects, f, indent=2)
    
    print(f"Converted to JSON array format: {output_file}")
    return len(objects)

def main():
    parser = argparse.ArgumentParser(description='Convert all JSONL files in needles directory to JSON format')
    parser.add_argument('--input-dir', default='needles', help='Input directory containing JSONL files')
    parser.add_argument('--output-dir', default='new_needles', help='Output directory for JSON files')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Find all JSONL files
    jsonl_files = list(input_dir.rglob('*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to convert")
    
    total_converted = 0
    for jsonl_file in jsonl_files:
        # Create output path by replacing input_dir with output_dir and .jsonl with .json
        relative_path = jsonl_file.relative_to(input_dir)
        output_file = output_dir / relative_path.with_suffix('.json')
        
        print(f"\nConverting: {jsonl_file} -> {output_file}")
        try:
            count = convert_jsonl_to_json(str(jsonl_file), str(output_file))
            total_converted += count
            print(f"✓ Converted {count} objects")
        except Exception as e:
            print(f"✗ Error converting {jsonl_file}: {e}")
    
    print(f"\nConversion complete! Total objects converted: {total_converted}")

if __name__ == "__main__":
    main()
