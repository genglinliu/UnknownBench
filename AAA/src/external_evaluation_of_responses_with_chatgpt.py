
# we are going to take the outputs of each LLM
# and evaluate the response text with the chatgpt model
# rate on the scale of 1-5, verbalized uncertainty. because we know chatgpt gives reliable verbalized uncertainty

import json
import os
import time
import openai
from prompts import experiment_prompts
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
import argparse

# global variables: model_list, task_list
openai_api_key = os.environ.get("OPENAI_API_KEY_HAO")

openai.api_key = openai_api_key

model_list = [
    'chatgpt',
    'palm',
    'claude',
    "Llama-2-7b-chat-hf",
    "Llama-2-7b-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-13b-hf",
    "Llama-2-70b-chat-hf",
    "Llama-2-70b-hf",
    "vicuna-7b-v1.5",
    "vicuna-13b-v1.5"
]

task_list = [
    "RefuNQ_unanswerable",
    "RefuNQ_answerable",
    "FalseQA_unanswerable",
    "FalseQA_answerable",
    "NEC_unanswerable",
    "NEC_answerable",
]

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

def evaluate_response_with_chatgpt(response) -> int:
    """
    Given a response, evaluate the response with chatgpt
    and get the verbalized uncertainty
    """
    prompt = experiment_prompts.prompt_verbalized_uncertainty_for_response.format(response)

    completion = openai.ChatCompletion.create(
        model=ENGINE,
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=64,
    )
    rating = completion['choices'][0]["message"]["content"]
    
    if not rating: # if null
        rating = 1
    elif "1" in rating:
        rating = 1
    elif "2" in rating:
        rating = 2
    elif "3" in rating:
        rating = 3
    elif "4" in rating:
        rating = 4
    elif "5" in rating:
        rating = 5
    else:
        rating = 1 # all other cases, treat as 1

    return rating

# Error callback function
def log_retry_error(retry_state):  
    print(f"Retrying due to error: {retry_state.outcome.exception()}")  
    
    
# @retry(wait=wait_random_exponential(min=6, max=60), stop=stop_after_attempt(80), retry_error_callback=log_retry_error)
def run(path_in, path_out, model_name):
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)

    with open(path_in, "r") as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        line = json.loads(line)
        if line["prompt"] in existing_samples_in_output_file:
            continue
        
        if model_name in ['chatgpt', 'palm', 'claude']:
            response = line[model_name] 
        else:
            response = line[f"{model_name}_response"]
        
        # now evaluate the response with chatgpt
        # and get the verbalized uncertainty
        eval_rating = evaluate_response_with_chatgpt(response)
        
        # put res into the json file
        line[f"rating_{model_name}"] = eval_rating
        # write the json file
        with open(path_out, 'a') as f:
            json.dump(line, f)
            f.write('\n')
            
        time.sleep(8) # sleep for 0.5 seconds to avoid openai api limit
        
        
def get_acc_vs_rating(path_out, model_name):
    """
    Given the output file, get the accuracy vs rating
    """
    with open(path_out, 'r') as f:
        lines = f.readlines()
        
    acc_vs_rating = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0], 5: [0, 0, 0]} # rating -> [correct, total, accuracy]
    for line in lines:
        data = json.loads(line)
        rating = data[f"rating_{model_name}"]
        label = data['label']
        
        if model_name in ['chatgpt', 'palm', 'claude']:
            response = data[model_name]
        else:
            response = data[f"{model_name}_response"]
        
        if not response:
            continue
        # if response contains any of the labels, then it is correct
        if any([x in response for x in label]):
            acc_vs_rating[rating][0] += 1
        acc_vs_rating[rating][1] += 1
        
    # get the average accuracy for each rating
    for rating in acc_vs_rating:
        if acc_vs_rating[rating][1] != 0:
            # round to 3 decimal places
            acc_vs_rating[rating][2] = round(acc_vs_rating[rating][0] / acc_vs_rating[rating][1], 3)    
        
        else:
            acc_vs_rating[rating][2] = 0
       
    return acc_vs_rating


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=model_list, required=True)
    parser.add_argument('--task', type=str, choices=task_list, required=True)
    parser.add_argument('--engine', type=str, choices=['gpt-3.5-turbo', 'gpt-4'], required=True)
    
    return parser.parse_args()
        

if __name__ == "__main__":
    args = vars(parse_args())
    
    task_name = args['task']
    model_name = args['model']
    if model_name in ['chatgpt', 'palm', 'claude']:
        path_in = f"AAA/outputs/Close-source-LLMs_responses/{model_name}/{model_name}_response_{task_name}.json"
    else:
        path_in = f"AAA/outputs/Llama-2_responses/{model_name}/{model_name}_{task_name}.json"
    
    ENGINE = args['engine']
    if ENGINE == 'gpt-3.5-turbo':
        path_out = f"/AAA/outputs/GPT-3.5-turbo_evals/"
    elif ENGINE == 'gpt-4':
        path_out = f"/AAA/outputs/GPT-4_evals/"
    
    # make path_out if not exist
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        
    path_out = os.path.join(path_out, f"{model_name}_{task_name}_output_confidence.json")
    
    run(path_in, path_out, model_name)
    
    # # evaluate the accuracy vs rating for every model
    # task_name = 'RefuNQ_answerable'
    # for model_name in model_list:
    #     print(model_name)
    #     path_out = f"/AAA/outputs/GPT-4_evals/"
    #     path_out = os.path.join(path_out, f"{model_name}_{task_name}_output_confidence.json")
    
    #     acc_vs_rating = get_acc_vs_rating(path_out, model_name)
    #     print(acc_vs_rating, '\n')
    

# python src/external_evaluation_of_responses_with_chatgpt.py --model Llama-2-7b-chat-hf --task RefuNQ_answerable --engine gpt-3.5-turbo