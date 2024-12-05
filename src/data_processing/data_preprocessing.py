# src/data_processing/data_preprocessing.py
import os
import re
import json

def extract_metadata(file_path):
    """
    Extracts metadata (CWE, description, source, sink, and functions) from the file's comments.
    """
    metadata = {
        "cwe": None,
        "description": None,
        "source": None,
        "sink": None,
        "functions": {"good": [], "bad": []}
    }

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

            # Extract CWE ID from the filename or comments
            match_cwe = re.search(r'Filename: (CWE\d+)', content)
            if match_cwe:
                metadata["cwe"] = match_cwe.group(1)

            # Extract description from comments
            match_description = re.search(r'@description\s+(.*)', content)
            if match_description:
                metadata["description"] = match_description.group(1).strip()

            # Extract source and sink information
            match_source = re.search(r'BadSource:\s+(.*)', content)
            if match_source:
                metadata["source"] = match_source.group(1).strip()
            match_sink = re.search(r'Sinks?:\s+(.*)', content)
            if match_sink:
                metadata["sink"] = match_sink.group(1).strip()

            # Identify good and bad functions
            metadata["functions"]["good"] = re.findall(r'(void\s+good\w*\s*\(\))', content)
            metadata["functions"]["bad"] = re.findall(r'(void\s+bad\w*\s*\(\))', content)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        metadata["error"] = str(e)

    return metadata

def process_files(base_path, output_file):
    """
    Traverses the directory, extracts metadata from each file, and saves it to a JSON file.
    """
    data = {}

    if not os.path.exists(base_path):
        print(f"Error: The specified path does not exist: {base_path}")
        return

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.c') or file.endswith('.cpp'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")  # Debugging output
                metadata = extract_metadata(file_path)
                data[file] = {
                    "file_path": file_path,
                    "metadata": metadata
                }

    # Save the structured data to JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Processed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving the output file: {e}")

if __name__ == "__main__":

    base_path = 'venv/data/raw/C/testcases'
    output_file = 'venv/raw/file_list.json'

    process_files(base_path, output_file)
