import json
import os
import argparse
import openai
from tqdm import tqdm
import time
import prompts.evaluation_prompts as eval_prompts
import re

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# openai.api_key = os.environ["OPENAI_API_KEY_HAO"]

SLEEP_RATE = 1.5 # sleep between calls
run_eval_prompt_on = eval_prompts.evaluation_prompt_version2

def get_existing_samples_in_output_file(path_out) -> set:
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

# @retry(wait=wait_random_exponential(min=30, max=300), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def evaluate():
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)

    with open(path_in_baseline, "r") as f, open(path_in_uncertainty, "r") as f_1:
        lines = f.readlines() 
        lines_uncertainty = f_1.readlines()
        
        for i in tqdm(range(len(lines))):
            line = json.loads(lines[i])
            line_uncertainty = json.loads(lines_uncertainty[i])
            
            if line["prompt"] in existing_samples_in_output_file: # skip negative samples and samples that are already in the output file
                continue

            # get model and output
            if 'gpt-3.5-turbo-0613' in line:
                response = line['gpt-3.5-turbo-0613']
                uncertainty = line_uncertainty['gpt-3.5-turbo-0613']
            elif 'claude-2' in line:
                response = line['claude-2']
                uncertainty = line_uncertainty['claude-2']
            elif 'palm-chat' in line:
                response = line['palm-chat']
                uncertainty = line_uncertainty['palm-chat']
            elif 'Llama-2-13b-chat-hf' in line:
                response = line['Llama-2-13b-chat-hf']
                uncertainty = line_uncertainty['Llama-2-13b-chat-hf']

            # evaluate
            res = "0"
            if response: 
                keywords = {"not", "no", "sorry", "don't", "can't", "won't", "mistake"}
                for target_word in keywords:
                    negative_keyword_present = re.search(r'\b' + target_word + r'\b', response.lower())
                    if negative_keyword_present:
                        res = "1"
                        
                    # special case 1 for NEC: endangered species
                    if negative_keyword_present and "endangered" in response.lower():
                        res = "0"
                        break
                    # special case 2 for NEC: significant role in ecosystem
                    if negative_keyword_present and "ecosystem" in response.lower():
                        res = "0"
                        break
                    # special cases for FalseQA: "which" or "don't"
                    if negative_keyword_present and "which" in response.lower():
                        res = "0"
                        break
                    if negative_keyword_present and "don't" in response.lower():
                        res = "0"
                        break
            
             
            eval_res = True if res == "1" else False # True == Refusal, False == No Refusal
           
            line["eval_res"] = eval_res
            line['confidence'] = uncertainty # change the name of the key to 'confidence', i.e. "1" is the lowest confidence
            with open(path_out, "a") as f:
                f.write(json.dumps(line) + "\n")
                
            # time.sleep(SLEEP_RATE) # sleep between calls


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['NEC', 'FalseQA'], required=True)
    parser.add_argument('--model', type=str, choices=['chatgpt', 'palm', 'llama2', 'claude'], required=True)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    MODE = "existent" # "existent" or "nonexistent"
    for TASK_NAME in ['NEC', 'FalseQA']:
        for MODEL_NAME in ['chatgpt', 'palm2', 'claude2']:
            
            if MODE == "nonexistent":
                path_in_baseline = f"/experiment_outputs/{TASK_NAME}/{MODEL_NAME}_baseline_{TASK_NAME}_out.json"
                path_in_uncertainty = f"/experiment_outputs/{TASK_NAME}/{MODEL_NAME}_uncertainty_{TASK_NAME}_out.json"
                path_out = f"/experiment_outputs/{TASK_NAME}/EVAL_{MODEL_NAME}_baseline_{TASK_NAME}.json"
            
            elif MODE == "existent":
                path_in_baseline = f"/experiment_outputs/ExistentConcepts/{MODEL_NAME}_baseline_{TASK_NAME}_out.json"
                path_in_uncertainty = f"/experiment_outputs/ExistentConcepts/{MODEL_NAME}_uncertainty_{TASK_NAME}_out.json"
                path_out = f"/experiment_outputs/ExistentConcepts/EVAL_{MODEL_NAME}_baseline_{TASK_NAME}.json"

            evaluate()
            time.sleep(1)