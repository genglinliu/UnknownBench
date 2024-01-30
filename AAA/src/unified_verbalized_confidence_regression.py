# this script is now the main engine that runs the experiments: both get responses and verbalized confidence on every model

import re
import time
import numpy as np
import argparse
import openai
import torch
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from prompts import experiment_prompts

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


import google.generativeai as palm

# API keys
PALM_API_KEY = os.environ.get("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY_HAO")

# other parameters
MAX_OUT_TOKENS = 32


def get_paths(model_name, task_name, task_partition, prompt_type):
    """
    tasks: RefuNQ, FalseQA, NEC
    task_partition: answerable, unanswerable

    return: 
        exp_name: e.g. Llama-2-7b-chat-hf_RefuNQ_answerable \\
        model_path, e.g. /mnt/data/models/Llama-2-7b-chat-hf \\
        data_path, e.g. data/NaturalQuestions/RefuNQ_2200.json \\
        output_path, e.g. experiment_outputs/Logits/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf_RefuNQ_answerable.json
    """
    data_path = f"AAA/data/{task_name}/{task_name}_{task_partition}.json"     
    exp_name = model_name + "_" + task_name + "_" + task_partition
    
    if prompt_type == "get_answer":
        output_path = f"AAA/outputs/UNIFIED_QA_responses/{model_name}"
    elif prompt_type == "get_confidence":
        output_path = f"experiment_outputs/UNIFIED_verbalize_confidence_no_ICL/{model_name}" 
    
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


# Error callback function
def log_retry_error(retry_state):  
    print(f"Retrying due to error: {retry_state.outcome.exception()}") 
    
    
def get_prompt(line, model_name, prompt_type=None) -> str:
    """
    return the formatted prompt string
    
    prompt_type: "get_answer" or "get_confidence"
    """
    if prompt_type == "get_answer":
        # same prompt for all models
        return experiment_prompts.prompt_baseline_chat.format(line)
    elif prompt_type == "get_confidence":
        # llama 2 completion models
        if model_name in ["Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf"]:
            return experiment_prompts.prompt_verbalized_uncertainty_for_input_regression_opensource_completion.format(line)
        # for all other models, use the same prompt
        else:
            return experiment_prompts.prompt_verbalized_uncertainty_for_input_regression_no_ICL.format(line)
    


def extract_number_from_response(res_raw: str, model_name: str) -> int:
    """
    extract the number from the response
    
    Do it differently for completion models and chat models
    """
    if res_raw is None:
        return None
    match = re.search(r'(\d+)', res_raw)
    res_numerical = int(match.group(1)) if match else None
    return res_numerical
    

# run chatgpt through api
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_chatgpt_through_openai_api(prompt: str, model_name: str) -> str:
    if model_name == "chatgpt":
        _model_name = "gpt-3.5-turbo-0613"
    elif model_name == "gpt-4-0613":
        _model_name = "gpt-4-0613"
    
    openai.api_key = OPENAI_API_KEY
    completion = openai.ChatCompletion.create(
        model=_model_name,
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=MAX_OUT_TOKENS,
    )
    res = completion['choices'][0]["message"]["content"]
    return res


# run claude2 through api
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_claude2_through_anthropic_api(prompt: str) -> str:
    anthropic = Anthropic(api_key=ANTHROPIC_API_KEY,)
    completion = anthropic.completions.create(
        model="claude-2",
        temperature=0,
        max_tokens_to_sample=MAX_OUT_TOKENS,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )
    res = completion.completion
    return res


# run chat-bison through api
palm.configure(api_key=PALM_API_KEY)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_chat_bison_through_palm_api(prompt: str) -> str:
    defaults = {
        'model': 'models/chat-bison-001',
        'temperature': 0
    }
    response = palm.chat(
        **defaults,
        context="Keep your response as short as possible. Strictly follow the instruction and do not output anything extra.",
        examples=[],
        messages=prompt
    )
    res = response.last # Response of the AI to your most recent request
    return res


# run llama2 or vicuna through api
def run_llama_or_vicuna_through_openai_api(prompt: str, model_name: str, url: str) -> str:
    openai.api_key = "EMPTY"
    openai.api_base = url
    
    if ('chat' in model_name) or ('vicuna' in model_name):
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=MAX_OUT_TOKENS,
        )
        res = completion['choices'][0]["message"]["content"]
    else:
        stop_tokens = ['Question', '\n\n', '\nAnswer'] 
        
        completion = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            temperature=0,
            max_tokens=MAX_OUT_TOKENS,
            stop=stop_tokens
        )
        res = completion['choices'][0]["text"]
        
    return res


def run_specific_model(model_name, prompt, url) -> str:
    """
    For every model_name we would use a different function that actually runs it
    output: response
    """
    if model_name == "chatgpt":
        response = run_chatgpt_through_openai_api(prompt, model_name)
    elif model_name == "gpt-4-0613":
        response = run_chatgpt_through_openai_api(prompt, model_name)
    elif model_name == "claude":
        response = run_claude2_through_anthropic_api(prompt)
    elif model_name == "palm":
        response = run_chat_bison_through_palm_api(prompt)
    elif ("Llama" in model_name) or ("vicuna" in model_name):
        response = run_llama_or_vicuna_through_openai_api(prompt, model_name, url)
        
    return response

# template to run any LLM on a single given prompt
def run_LLM(path_in, path_out, model_name, task_partition, url, prompt_type):
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    
    with open(path_in, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        line = json.loads(line)
        
        # skip if the sample already exists in the output file
        if  (line["prompt"] in existing_samples_in_output_file) or \
            (line["label"] == 0 and task_partition == "unanswerable") or \
            (line["label"] == 1 and task_partition == "answerable"):
            continue
        
        prompt = get_prompt(line["prompt"], model_name, prompt_type)        
        res_raw = run_specific_model(model_name, prompt, url)
        
        # extract the number from res_raw, if not found then return None
        res_numerical = extract_number_from_response(res_raw, model_name)

        # put res into the json file
        line[model_name] = res_raw
        
        if prompt_type == "get_confidence":
            line[model_name + "_numerical"] = res_numerical
        
        # write the json file
        with open(path_out, 'a') as f:
            json.dump(line, f)
            f.write('\n')        
        
# main 
if __name__ == "__main__":
    
    proprietary_model_list = [
        "chatgpt", # gpt-3.5-turbo-0613
        "gpt-4-0613",
        "claude",
        "palm"
    ]
    
    opensource_model_list = [
        "Llama-2-7b-chat-hf",
        "Llama-2-7b-hf",
        "Llama-2-13b-chat-hf",
        "Llama-2-13b-hf",
        "Llama-2-70b-chat-hf",
        "Llama-2-70b-hf",
        "vicuna-7b-v1.5",
        "vicuna-13b-v1.5"
    ]
    
    model_list_all = proprietary_model_list + opensource_model_list
    
    task_names = ["RefuNQ", "FalseQA", "NEC"]
    task_partitions = ["answerable", "unanswerable"]
    
    # get argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=model_list_all, help="The name of the model")
    parser.add_argument("--prompt_type", type=str, choices=["get_answer", "get_confidence"], help="The type of prompt")
    args = parser.parse_args()
    
    model_name = args.model_name
    prompt_type = args.prompt_type

    url = 'http://localhost:8000/v1' # workstation vllm server
        
    torch.cuda.empty_cache()
    for task_name in task_names:
        for task_partition in task_partitions:
            print("Currently running: ", model_name, task_name, task_partition)
            print("=========================================")
            
            exp_name, data_path, output_path = get_paths(model_name, task_name, task_partition, prompt_type)
            
            run_LLM(data_path, output_path, model_name, task_partition, url, prompt_type)

