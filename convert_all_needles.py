#!/usr/bin/env python3
"""
Script to convert all JSONL files in the needles folder to JSON format in new_needles folder
"""
import os
import subprocess
import sys
from pathlib import Path

def find_jsonl_files(root_dir):
    """Find all JSONL files in the directory tree"""
    jsonl_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def create_output_structure(jsonl_files, source_root, target_root):
    """Create the target directory structure and return mapping of input to output files"""
    file_mappings = []
    
    for jsonl_file in jsonl_files:
        # Get relative path from source root
        rel_path = os.path.relpath(jsonl_file, source_root)
        
        # Create output path by replacing .jsonl with .json
        output_rel_path = rel_path[:-6] + '.json'  # Remove .jsonl and add .json
        output_file = os.path.join(target_root, output_rel_path)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        file_mappings.append((jsonl_file, output_file))
    
    return file_mappings

def convert_jsonl_to_json(input_file, output_file):
    """Convert a single JSONL file to JSON using the jsonl_to_json.py script"""
    try:
        # Run the jsonl_to_json.py script
        result = subprocess.run([
            sys.executable, 'jsonl_to_json.py', input_file, '-o', output_file
        ], capture_output=True, text=True, check=True)
        
        print(f"✓ Converted: {input_file} -> {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting {input_file}: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error converting {input_file}: {e}")
        return False

def main():
    source_root = "needles"
    target_root = "new_needles"
    
    print(f"Converting all JSONL files from {source_root} to {target_root}")
    print("=" * 60)
    
    # Find all JSONL files
    jsonl_files = find_jsonl_files(source_root)
    print(f"Found {len(jsonl_files)} JSONL files to convert")
    
    if not jsonl_files:
        print("No JSONL files found!")
        return
    
    # Create output structure and get file mappings
    file_mappings = create_output_structure(jsonl_files, source_root, target_root)
    
    # Convert each file
    successful_conversions = 0
    failed_conversions = 0
    
    for input_file, output_file in file_mappings:
        if convert_jsonl_to_json(input_file, output_file):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    print("=" * 60)
    print(f"Conversion complete!")
    print(f"✓ Successfully converted: {successful_conversions} files")
    print(f"✗ Failed conversions: {failed_conversions} files")
    
    if failed_conversions == 0:
        print("All conversions successful! 🎉")
    else:
        print(f"Some conversions failed. Please check the errors above.")

if __name__ == "__main__":
    main()






