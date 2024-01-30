import argparse
import logging
import os
import openai
import replicate
import json
from tqdm import tqdm
from prompts import experiment_prompts as exp_prompts
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import requests
from time import sleep

# EXP_NAME = "9_4_claude_existent"

# API keys
openai.api_key = os.environ.get("OPENAI_API_KEY_HAO")
replicate_api_key = os.environ.get("REPLICATE_API_KEY") # for Vicuna-13B
bard_key = os.environ.get("GOOGLE_API_KEY")

# verbalized check prompt
prompt_baseline = exp_prompts.prompt_baseline
prompt_verbalized_check = exp_prompts.prompt_verbalized_check
prompt_verbalized_uncertainty = exp_prompts.prompt_verbalized_uncertainty
# prompt_verbalized_uncertainty_llama = exp_prompts.prompt_verbalized_uncertainty_LLAMA
prompt_verbalized_uncertainty_v2 = exp_prompts.prompt_verbalized_uncertainty_v2

# PaperCitation specific prompt
prompt_paper_citation = exp_prompts.prompt_paper_citation
prompt_paper_citation_with_checks = exp_prompts.prompt_paper_citation_with_checks


def get_prompt(task_type, line) -> str:
    """
    PROMPT_TYPE: "baseline" or "uncertainty" "verbalized_check"
    task_type: "PaperCitation" or "NEC" or "FalseQA"
    line: the line["prompt"] from the json file
    
    return:
    the formatted prompt string
    """
    if PROMPT_TYPE == "baseline":
        if 'llama' in MODEL and 'chat' not in MODEL:
            # completion model
            # print("using special prompt for llama completion model")
            return exp_prompts.prompt_baseline_for_completion.format(line)
        
        # if task_type is "paper_citation", then we need to format the `prompt_paper_citation` with the line
        if task_type == "PaperCitation":
            return prompt_paper_citation.format(line)
        # otherwise we just need to format the `prompt_baseline` with the line
        else:
            return exp_prompts.prompt_baseline.format(line)
        
    elif PROMPT_TYPE == "uncertainty":
        if 'llama' in MODEL and 'chat' not in MODEL:
            return exp_prompts.prompt_verbalized_uncertainty_completion.format(line)
        else:
            return exp_prompts.prompt_verbalized_uncertainty_v2.format(line)
    if PROMPT_TYPE == "combined":
        if task_type == "PaperCitation":
            # if task_type is "paper_citation", we first format the `prompt_paper_citation` with the line as an intermediate result
            return prompt_paper_citation_with_checks.format(line)
        else:
            return prompt_verbalized_check.format(line)

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


# OpenAI api: ChatGPT (gpt3.5) / GPT-4
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_GPT_chat(path_in, path_out, task_type, model="gpt-3.5-turbo"):
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    
    with open(path_in, 'r') as f:
        lines = f.readlines()
        # lines = lines[START:END]
        for line in tqdm(lines):
            line = json.loads(line)
            
            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
           
            
            prompt = get_prompt(task_type, line["prompt"])
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=MAX_OUT_TOKENS,
            )
            res = completion['choices'][0]["message"]["content"]
            # put res into the json file
            line[model] = res
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
            sleep(1)

# Palm api - chat-bison
# code template source: https://makersuite.google.com/app/prompts/new_multiturn

import google.generativeai as palm

palm.configure(api_key=bard_key)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_palm_chat(path_in, path_out, task_type):
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()
        # lines = lines[START:END]
        for line in tqdm(lines):
            line = json.loads(line)
            
            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            
            defaults = {
                'model': 'models/chat-bison-001',
                'temperature': 0,
                'candidate_count': 1,
                'top_k': 40,
                'top_p': 1
                # 'output_token_limit': MAX_OUT_TOKENS
            }
            prompt = get_prompt(task_type, line["prompt"])
            response = palm.chat(
                **defaults,
                context="Answer the question and be as short and concise as possible. Keep the response to one sentence.",
                examples=[],
                messages=prompt
            )
            # print(response.last) 
            # break
            res = response.last # Response of the AI to your most recent request
            # put res into the json file
            line["palm-chat"] = res
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
    
# Vicuna-13B api
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), retry_error_callback=log_retry_error)
def run_Vicuna13B(path_in, path_out, task_type):
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()
        # lines = lines[START:END]
        for line in tqdm(lines):
            line = json.loads(line)
            
            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
                
            replicate_client = replicate.Client(api_token=replicate_api_key)
            prompt = get_prompt(task_type, line["prompt"])
            input_params = {
                "prompt": prompt,
                "temperature": 0.1,
                "max_length": MAX_OUT_TOKENS,
                "top_p": 1,
                "repitition_penalty": 1,
            }
            output = replicate_client.run(
                "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
                input=input_params,
            )

            # The predict method returns an iterator (output), res is a string joined from that iterator
            res = "".join(output)

            # # put res into the json file
            line["vicuna-13b"] = res
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
                
                
# Llama-2 api - replicate
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_Llama2(path_in, path_out, task_type):
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()
        # lines = lines[START:END]
        for line in tqdm(lines):
            line = json.loads(line)

            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
                
            replicate_client = replicate.Client(api_token=replicate_api_key)
            prompt = get_prompt(task_type, line["prompt"])
            input_params = {
                "prompt": prompt,
                "system_prompt": "Respond in the precise format requested by the user. Do not acknowledge requests with 'sure' or in any other way besides going straight to the answer.",
                "temperature": 0.1,
                "max_new_tokens": MAX_OUT_TOKENS,
                "min_new_tokens": 2,
                "top_p": 1,
            }
            output = replicate_client.run(
                "a16z-infra/llama-2-13b-chat:d5da4236b006f967ceb7da037be9cfc3924b20d21fed88e1e94f19d56e2d3111",
                input=input_params,
            )

            # The predict method returns an iterator (output), res is a string joined from that iterator
            res = "".join(output)

            # # put res into the json file
            line["llama-2-13b"] = res
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
                
                
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)      
def run_llama2_blender(path_in, path_out, task_type, model, url):
    sys_msg="You are a helpful assistant."
    alt_sys_msg = "Answer the given question in no more than one sentence. Keep your answer short and concise."
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            prompt = get_prompt(task_type, line["prompt"])

            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' 
            }

            data = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': sys_msg}, 
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0,
                'max_tokens': MAX_OUT_TOKENS
            }
            
            response = requests.post(url, headers=headers, json=data)       
            res = response.json()
            # print(res)
            # break     
            res = response.json()['choices'][0]['message']['content']

            # put res into the json file
            line[model] = res
            
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
                

def run_llama_through_openai_api(path_in, path_out, task_type, model, url):
    openai.api_key = "EMPTY"
    openai.api_base = url
    
    print(openai.api_base)
    
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    
    with open(path_in, 'r') as f:
        lines = f.readlines()
        # lines = lines[START:END]
        for line in tqdm(lines):
            line = json.loads(line)
            
            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
           
            
            prompt = get_prompt(task_type, line["prompt"])
            
            
            if 'chat' in model:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=MAX_OUT_TOKENS,
                )
                res = completion['choices'][0]["message"]["content"]
            else:
                completion = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=MAX_OUT_TOKENS,
                    # stop
                    stop=['Question', '\n\n', '\nAnswer']
                )
                res = completion['choices'][0]["text"]
            
            # put res into the json file
            line[model] = res
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
                
                
# run Claude
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_claude(path_in, path_out, task_type):
    
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = json.loads(line)
        
        if MODE == "existent":
            if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                continue
        elif MODE == "nonexistent":
            if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                continue
            
        prompt = get_prompt(task_type, line["prompt"])
        
        anthropic = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=128,
            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
        )
        
        # put res into the json file
        line['claude-2'] = completion.completion
        
        # write the json file
        with open(path_out, 'a') as f:
            json.dump(line, f)
            f.write('\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, choices=['baseline', 'uncertainty'], required=True)
    parser.add_argument('--model', type=str, choices=['chatgpt', 'palm2', 'llama2','llama2-7b', 'llama2-13b-completion', 'llama2-70b', 'llama2-7b-chat', 'llama2-7b-completion', 'llama2-70b-chat', 'llama2-70b-completion', 'claude2'], required=True)
    
    return parser.parse_args()

# main function
if __name__ == "__main__":
    
    args = vars(parse_args())
    PROMPT_TYPE = args['prompt']
    MODEL = args['model']
    
    MAX_OUT_TOKENS = 128
    
    MODE = "nonexistent" # "existent" or "nonexistent"
    
    URL = "" # url for llama 2 server
    

    if MODE == "nonexistent":
        # data paths
        NEC_path_in = "data/NonExsitentConcept/NEC_data_new_with_categories.json"
        FalseQA_path_in = "data/FalseQA/FalseQA_data_new.json"
        NQ_path_in = "data/NaturalQuestions/RefuNQ_2200.json"
        # experiment results paths
        NEC_path_out = f"experiment_outputs/NEC/{MODEL}_{PROMPT_TYPE}_NEC_out.json"
        FalseQA_path_out = f"experiment_outputs/FalseQA/{MODEL}_{PROMPT_TYPE}_FalseQA_out.json"
        NQ_path_out = f"experiment_outputs/RefuNQ/{MODEL}_{PROMPT_TYPE}_RefuNQ_out.json"
        
    elif MODE == "existent":
        # data paths
        NEC_path_in = "data/NonExsitentConcept/EC_data_with_categories.json"
        FalseQA_path_in = "data/FalseQA/FalseQA_data_normal_questions.json"
        NQ_path_in = "data/NaturalQuestions/NQ_2200_filtered.json"
        # experiment results paths
        NEC_path_out = f"experiment_outputs/ExistentConcepts/{MODEL}_{PROMPT_TYPE}_NEC_out.json"
        FalseQA_path_out = f"experiment_outputs/ExistentConcepts/{MODEL}_{PROMPT_TYPE}_FalseQA_out.json"
        NQ_path_out = f"experiment_outputs/NQ/{MODEL}_{PROMPT_TYPE}_NQ_out.json"

    # run the models
    if MODEL == 'chatgpt':
        print("Running gpt-3.5-turbo-0613...")
        run_GPT_chat(NQ_path_in, NQ_path_out, task_type="NQ", model="gpt-3.5-turbo-0613")
    elif MODEL == 'palm2':
        print("Running chat-bison...")
        run_palm_chat(NQ_path_in, NQ_path_out, task_type="NQ")
    elif 'llama2' in MODEL: # llama 2 family
        print("Running Llama-2...")
        if MODEL == 'llama2': # llama-2-13b-chat-hf
            run_llama_through_openai_api(NQ_path_in, NQ_path_out, task_type="NQ", model="Llama-2-13b-chat-hf", url=URL)
    elif MODEL == 'claude2':
        print("Running claude-2...")
        run_claude(NQ_path_in, NQ_path_out, task_type="NQ")