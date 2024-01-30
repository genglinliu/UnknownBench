import json
import re

def extract_confidence(json_line, model_name):
    # Extract the number inside [] using regex
    if json_line.get(model_name) is None:
        return None
    match = re.search(r'\[(\d+)\]', json_line.get(model_name, ''))
    # If found, return the number, otherwise return None
    return int(match.group(1)) if match else None

def process_file(input_file, output_file, model_name):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for data in infile:
            line = json.loads(data)
            confidence = extract_confidence(line, model_name)
            
            line[f"{model_name}_numerical"] = confidence                
            outfile.write(json.dumps(line) + '\n')


models = ["chatgpt", "claude", "palm"]
tasks = ["RefuNQ", "FalseQA", "NEC"]
modes = ["answerable", "unanswerable"]

for model_name in models:
    for task in tasks:
        for mode in modes:
            input_file = f"/experiment_outputs/UNIFIED_verbalize_confidence/{model_name}/{model_name}_{task}_{mode}.json"
            output_file = f"/experiment_outputs/UNIFIED_verbalize_confidence/extracted_scores/{model_name}/{model_name}_{task}_{mode}.json"

            process_file(input_file, output_file, model_name)
