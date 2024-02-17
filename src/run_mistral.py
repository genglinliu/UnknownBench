import time
import openai
import replicate
import numpy as np
import argparse
import requests
import torch
import json
import os
from tqdm import tqdm
from AAA.prompts import experiment_prompts as exp_prompts


def get_paths(model_name, task_name, mode):
    """
    tasks: RefuNQ, FalseQA, NEC
    mode: answerable, unanswerable

    return: 
        exp_name: e.g. Llama-2-7b-chat-hf_RefuNQ_answerable \\
        model_path, e.g. /mnt/data/models/Llama-2-7b-chat-hf \\
        data_path, e.g. data/NaturalQuestions/RefuNQ_2200.json \\
        output_path, e.g. experiment_outputs/Logits/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf_RefuNQ_answerable.json
    """

    
    data_path = f"data/{task_name}/{task_name}_{mode}.json"

    exp_name = model_name + "_" + task_name + "_" + mode
    
    output_path = f"experiment_outputs/{model_name}"
    
    if GET_CONFIDENCE_SCORES:
        output_path = os.path.join(output_path, "confidence_scores")
    
    # create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_path = os.path.join(output_path, exp_name + ".json")
    
    print("Data path: ", data_path)
    print("Output path: ", output_path)
    
    return exp_name, data_path, output_path


def get_existing_samples_in_output_file(path_out) -> set:
    """
    If we need to rerun the script, we don't need to rerun the samples that are already in the output file
    """
    existing_samples_in_output_file = set()
    if not os.path.exists(path_out):
        # if the output file does not exist, then return an empty set
        return existing_samples_in_output_file
        
    with open(path_out, 'r') as f:
        lines_output = f.readlines()
        for line in lines_output:
            data = json.loads(line)
            if data['prompt'] not in existing_samples_in_output_file:
                existing_samples_in_output_file.add(data['prompt'])
    return existing_samples_in_output_file


    
def run_mistral_via_openai(prompt, model_name):
    model_name = "mistral"

    openai.api_key = 'ollama'
    openai.api_base = 'http://localhost:11434/v1'

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=64,
        stop=["\n\n"]
    )
    res = completion['choices'][0]["message"]["content"]
    return res

def run_mistral_base_via_request(prompt, model_name): # base model
    assert model_name == "mistral-base" # only the base model is supported  
    # Define the URL and the payload
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:text",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.01,
            # "stop": ["\n", "1"],
            'stop': ["\n"],
            "num_predict": 32
        }
    }
    
    response_text = requests.post(url, json=payload).text
    json_response = json.loads(response_text)
    return json_response["response"]



def run(path_in, path_out, model_name, mode):
    """
    Load the samples from the input file
    
    If SKIP_EXISTING_SAMPLES is True, then skip the samples that already exist in the output file
    If GET_LOGIT_BASED_UNCERTAINTY is True, then calculate the entropy for each sample in the input file
    If GET_RESPONSE_FROM_OPENAI_API is True, then get the response from the vllm OpenAI API for each sample in the input file
    """
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        line = json.loads(line)
        
        if SKIP_EXISTING_SAMPLES:
            # skip if the sample already exists in the output file
            if  (line["prompt"] in existing_samples_in_output_file) or \
                (line["label"] == 0 and mode == "unanswerable") or \
                (line["label"] == 1 and mode == "answerable"):
                continue
            
        if not GET_CONFIDENCE_SCORES:
            if "base" in model_name:
                prompt = exp_prompts.prompt_baseline_completion.format(line["prompt"])
            else:
                prompt = exp_prompts.prompt_baseline_chat.format(line["prompt"])
            
        if GET_CONFIDENCE_SCORES:
            prompt = exp_prompts.prompt_verbalized_uncertainty_no_IC.format(line["prompt"])   
        
        if "base" in model_name:
            res = run_mistral_base_via_request(prompt, model_name)
        else:    
            res = run_mistral_via_openai(prompt, model_name)
        
        line[f'{model_name}_response'] = res
        # write the json file
        with open(path_out, 'a') as f:
            json.dump(line, f)
            f.write('\n')
    

# main 
SKIP_EXISTING_SAMPLES = True
GET_CONFIDENCE_SCORES = True

if __name__ == "__main__":
    
    model_list = [
        "mistral",
        "mistral-base"
    ]
    
    tasks = ["RefuNQ", "FalseQA", "NEC"]
    modes = ["answerable", "unanswerable"]
    
    # get argpa
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=model_list, help="The name of the model")
    args = parser.parse_args()
    
    model_name = args.model_name
    
    torch.cuda.empty_cache()
    for task_name in tasks:
        for mode in modes:
            print("Model: ", model_name)
            print("Task: ", task_name)
            print("Mode: ", mode)
            print("=========================================")
            
            exp_name, data_path, output_path = get_paths(model_name, task_name, mode)
            
            run(data_path, output_path, model_name, mode)
            
            # clear cache
            torch.cuda.empty_cache()

        time.sleep(1)