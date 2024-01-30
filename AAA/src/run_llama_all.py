import re
import time
import accelerate
import numpy as np
import argparse
import openai
import torch
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from prompts import experiment_prompts as exp_prompts

from transformers import AutoTokenizer, AutoModelForCausalLM



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

    if task_name == 'RefuNQ' and mode == "unanswerable":
        data_path = "data/NaturalQuestions/RefuNQ_2200.json"
        
    elif task_name == 'RefuNQ' and mode == "answerable":
        data_path = "data/NaturalQuestions/NQ_2200_filtered.json"

    elif task_name == 'FalseQA' and mode == "unanswerable":
        data_path = "data/FalseQA/FalseQA_data_new.json"

    elif task_name == 'FalseQA' and mode == "answerable":
        data_path = "data/FalseQA/FalseQA_data_normal_questions.json"
        
    elif task_name == 'NEC' and mode == "unanswerable":
        data_path = "data/NonExsitentConcept/NEC_data_new_with_categories.json"

    elif task_name == 'NEC' and mode == "answerable":
        data_path = "data/NonExsitentConcept/EC_data_with_categories.json"

    exp_name = model_name + "_" + task_name + "_" + mode
    
    if GET_LOGIT_BASED_UNCERTAINTY:
        output_path = f"experiment_outputs/Logits-v3/{model_name}" 
    
    elif GET_RESPONSE_FROM_OPENAI_API:
        output_path = f"experiment_outputs/Llama_Responses/additional/{model_name}"
    
    # create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_path = os.path.join(output_path, exp_name + ".json")
    
    print("Data path: ", data_path)
    print("Output path: ", output_path)
    
    return exp_name, data_path, output_path


# load model
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token # </s> for llama tokenizer
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training, https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd#scrollTo=OJXpOgBFuSrc&line=34&uniqifier=1

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    print("Model loaded")
    return tokenizer, model


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


def calculate_entropy(input_ids) -> float:
    """
    Calculate the entropy of the next token distribution
    """
    with torch.no_grad():
        outputs = llama_model_object(input_ids)
        logits = outputs.logits # logits is a tensor of shape (batch_size, sequence_length, vocab_size)

    # Calculate the probabilities using the softmax function
    probabilities = torch.nn.functional.softmax(logits[0, -1, :], dim=0) # logits[0, -1, :] is the logits of the last token

    # Calculate the entropy of the next token distribution
    entropy = -torch.sum(probabilities * torch.log(probabilities))
    return entropy.item()


def calculate_perplexity(input_ids):
    """
    Calculate the perplexity of the whole input sequence
    
    documentation on the loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    
    perplexity = exp(cross_entropy_loss)
    """
    with torch.no_grad():
        loss = llama_model_object(input_ids, labels=input_ids).loss # this loss is the cross entropy loss, defined as the negative log likelihood
    
    perplexity = torch.exp(loss)
    return perplexity.item()

    
def run_llama_through_openai_api(prompt, model_name, url):
    openai.api_key = "EMPTY"
    openai.api_base = url
    
    MAX_OUT_TOKENS = 128

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
        completion = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            temperature=0,
            max_tokens=MAX_OUT_TOKENS,
            # stop
            stop=['Question', '\n\n', '\nAnswer']
        )
        res = completion['choices'][0]["text"]
        
    return res


def plot_entropy_and_perplexity(output_path, model_name, exp_name):
    """
    Plot a histogram of the entropy, perplexity, and log likelihood
    """
    save_figure_path = f"experiment_outputs/Figures/Logits-v3/{model_name}"
    if not os.path.exists(save_figure_path):
        os.makedirs(save_figure_path)
    
    entropies = []
    perplexities = []
    
    with open(output_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            entropies.append(line[f'{model_name}_entropy'])
            perplexities.append(line[f'{model_name}_perplexity'])
    
            
    # plot entropy
    plt.figure(figsize=(8, 6))
    plt.hist(entropies, bins=100, alpha=0.5, color='blue')
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.title(f"Entropy: {exp_name}")
    plt.savefig(save_figure_path + f"/{exp_name}_entropy.pdf")
        
    # plot perplexity
    plt.figure(figsize=(8, 6))
    plt.hist(perplexities, bins=100, alpha=0.5, color='orange')
    plt.xlabel("Perplexity")
    plt.ylabel("Frequency")
    plt.title(f"Perplexity: {exp_name}")
    plt.savefig(save_figure_path + f"/{exp_name}_perplexity.pdf")
    
    plt.close('all')
    

def run_llama2_model(path_in, path_out, model_name, mode, url):
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
        
        if GET_RESPONSE_FROM_OPENAI_API:
            # get prompt - no special tokens needed because fastchat takes care of it
            if ('chat' in model_name) or ('vicuna' in model_name):
                prompt = exp_prompts.prompt_baseline_chat_v2.format(line["prompt"])
            else:
                prompt = exp_prompts.prompt_baseline_completion_v2.format(line["prompt"])
            res = run_llama_through_openai_api(prompt, model_name, url)
            line[f'{model_name}_response'] = res
        
        if GET_LOGIT_BASED_UNCERTAINTY:
            if ('chat' in model_name) or ('vicuna' in model_name):
                prompt = exp_prompts.prompt_baseline_chat_llama_v2.format(line["prompt"])
            else:
                prompt = exp_prompts.prompt_baseline_completion_v2.format(line["prompt"])
            
            # get entropy
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            entropy = calculate_entropy(input_ids)
            perplexity = calculate_perplexity(input_ids)
            
            line[f'{model_name}_entropy'] = entropy
            line[f'{model_name}_perplexity'] = perplexity
        
        # write the json file
        with open(path_out, 'a') as f:
            json.dump(line, f)
            f.write('\n')

    if GET_LOGIT_BASED_UNCERTAINTY:
        plot_entropy_and_perplexity(output_path, model_name, exp_name)
    

# main 
SKIP_EXISTING_SAMPLES = True
GET_RESPONSE_FROM_OPENAI_API = True
RUN_SPECIFIC_MODEL = True
GET_LOGIT_BASED_UNCERTAINTY = not GET_RESPONSE_FROM_OPENAI_API # Only one of GET_RESPONSE_FROM_OPENAI_API and GET_LOGIT_BASED_UNCERTAINTY can be True
print("GET_RESPONSE_FROM_OPENAI_API: ", GET_RESPONSE_FROM_OPENAI_API)
print("GET_LOGIT_BASED_UNCERTAINTY: ", GET_LOGIT_BASED_UNCERTAINTY)

if __name__ == "__main__":
    
    llama_2_model_list = [
        "Llama-2-7b-chat-hf",
        "Llama-2-7b-hf",
        "Llama-2-13b-chat-hf",
        "Llama-2-13b-hf",
        "Llama-2-70b-chat-hf",
        "Llama-2-70b-hf",
        "vicuna-7b-v1.5",
        "vicuna-13b-v1.5"
    ]
    
    tasks = ["RefuNQ", "FalseQA", "NEC"]
    modes = ["answerable", "unanswerable"]
    
    # get argparse
    if RUN_SPECIFIC_MODEL:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, choices=llama_2_model_list, help="The name of the model")
        # parser.add_argument("--task_name", type=str, help="The name of the task")
        # parser.add_argument("--mode", type=str, help="The mode of the task")
        args = parser.parse_args()
        
        model_name = args.model_name
        if GET_LOGIT_BASED_UNCERTAINTY:
            model_path = f"/mnt/data/models/{model_name}"
            tokenizer, llama_model_object = load_model(model_path)
            url = None # we don't need the url if we are not using the OpenAI API
        if GET_RESPONSE_FROM_OPENAI_API:
            llama_model_object = None # we don't need to load the model if we are using the OpenAI API
            url = 'http://localhost:8000/v1' # 13b-hf
        
        torch.cuda.empty_cache()
        for task_name in tasks:
            for mode in modes:
                print("Model: ", model_name)
                print("Task: ", task_name)
                print("Mode: ", mode)
                print("=========================================")
                
                exp_name, data_path, output_path = get_paths(model_name, task_name, mode)
                
                run_llama2_model(data_path, output_path, model_name, mode, url)
                
                # clear cache
                torch.cuda.empty_cache()

            time.sleep(1)
    else:
        for model_name in llama_2_model_list:
            
            if GET_LOGIT_BASED_UNCERTAINTY:
                model_path = f"/mnt/data/models/{model_name}"
                tokenizer, llama_model_object = load_model(model_path)
                url = None # we don't need the url if we are not using the OpenAI API
            if GET_RESPONSE_FROM_OPENAI_API:
                llama_model_object = None # we don't need to load the model if we are using the OpenAI API
                # url = 'http://localhost:8000/v1' # 7b-hf
            
            torch.cuda.empty_cache()
            for task_name in tasks:
                for mode in modes:
                    print("Model: ", model_name)
                    print("Task: ", task_name)
                    print("Mode: ", mode)
                    print("=========================================")
                    
                    exp_name, data_path, output_path = get_paths(model_name, task_name, mode)
                    
                    run_llama2_model(data_path, output_path, model_name, mode, url)
                    
                    # clear cache
                    torch.cuda.empty_cache()

                time.sleep(1)
                    