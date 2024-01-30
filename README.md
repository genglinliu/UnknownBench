# README

This is the repository for the paper Revisit Hallucination in the Era of Large Language Models


## Setup for running the logit experiments on LLama-2 

 - install dependencies `pip install -r requirements.txt`
 - set pythonpath `export PYTHONPATH="${PYTHONPATH}:/path/to/LLM-hallucination/"`
 - run `python src/run_llama_all.py`


## datasets 

Tasks: `FalseQA`, `NEC`, `RefuNQ` (each task has two modes: `answerable` and `unanswerable`)

```python
data_path = f"AAA/data/{task_name}/{task_name}_{mode}.json"  
```

Results are stored at `experiment_outputs/Logits`
Figures are saved at `experiment_outputs/figures`


## Setup

- install dependencies
`pip install -r requirements.txt`

- set pythonpath
`export PYTHONPATH="${PYTHONPATH}:/path/to/LLM-hallucination/"`

i.e. `export PYTHONPATH="${PYTHONPATH}:/home/genglin2/LLM-hallucination"`


- set up your own OpenAI API, Google BARD API, Replicate API
`source api_key_config.sh`


- For claude, you may need to run `ulimit -n 2048` to prevent a potential `too many open files` error. 

## Quick Start

### generate LLM responses on FalseQA / NEC / PaperCitation

Our work contains three tasks:
 - FalseQA: answering questions that may contain false premises
 - Non-existent Concepts (NEC): explain questions that might involve nonexistent concepts
 - Paper Citation: given a paragraph of a paper, fill the blank with paper titles.

To run different models on these tasks, we have 

`python src/run.py --prompt baseline`

Notes for Xingyao on 8/18: 
 - Currently this script runs on NEC and FalseQA sequencially, and only with gpt-3.5. All other models and tasks are hidden. 
 - --prompt should just be followed by the `baseline` flag as other prompt checks are still under construction.
 - for now we can just generate responses and do eval later (with human eval too)

### Evaluation

To evaluate the outputs of the LLMs, run

`python scripts/evaluation/NEC_and_FalseQA_eval.py` for FalseQA and NEC.

`python scripts/evaluation/PaperCitation_SemanticScholar_verify.py` for PaperCitation

## Prompts
The prompts used in this repo can be found in the `prompts/` folder. 

## Datasets
All the benchmark data can be found in the `data/` folder.

## Citation

TBA
