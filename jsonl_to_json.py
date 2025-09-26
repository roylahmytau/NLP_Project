#!/usr/bin/env python3
"""
Script to convert JSONL format to JSON format by adding commas between objects
"""
import json
import argparse
import os

def jsonl_to_json(input_file, output_file):
    """Convert JSONL file to JSON array format by adding commas between objects"""
    objects = []
    
    with open(input_file, 'r') as f:
        content = f.read().strip()
    
    # Split by the pattern where one object ends and another begins
    # Look for } followed by {"index"
    import re
    # Split on the pattern: } followed by {"index"
    parts = re.split(r'{"index"', content)
    # remove the first
    parts = parts[1:]
    print(f"Split into {len(parts)} parts")
    # Process each part
    for i, part in enumerate(parts):
            
        # Add back the opening brace and index for all parts except the first
        part = '{"index"' + part
        # if all chars from the end until '}' is seen
        while not part.endswith('}'):
            part = part[:-1]

        # debug
        if i == 1:
            print(part)
        try:
            obj = json.loads(part)
            objects.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing part {i+1}: {e}")
            continue
    
    print(f"Successfully parsed {len(objects)} JSON objects from {input_file}")
    
    # Write as JSON array
    with open(output_file, 'w') as f:
        json.dump(objects, f, indent=2)
    
    print(f"Converted to JSON array format: {output_file}")
    return len(objects)

def main():
    parser = argparse.ArgumentParser(description='Convert JSONL format to JSON array format')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('-o', '--output', help='Output JSON file path (optional, defaults to input file with .json extension)')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    
    # Generate output file name if not provided
    if args.output:
        output_file = args.output
    else:
        # Replace .jsonl extension with .json
        if input_file.endswith('.jsonl'):
            output_file = input_file[:-6] + '.json'
        else:
            output_file = input_file + '.json'
    
    print(f"Converting {input_file} to JSON array format...")
    count = jsonl_to_json(input_file, output_file)
    print(f"Successfully processed {count} JSON objects")
    
    # Test the converted file
    print("\nTesting converted file...")
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
            print(f"JSON array contains {len(data)} objects")
            if data:
                print(f"First object keys: {list(data[0].keys())}")
                print(f"Expected keys: ['index', 'outputs', 'length', 'length_w_model_temp', 'answer_prefix', 'instruction', 'doc', 'question']")
    except Exception as e:
        print(f"Error testing converted file: {e}")

if __name__ == "__main__":
    main()