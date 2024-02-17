# [Examining LLMs' Uncertainty Expression Towards Questions Outside Parametric Knowledge](https://arxiv.org/abs/2311.09731)


## Setup

- install dependencies
`pip install -r requirements.txt`

- set pythonpath
`export PYTHONPATH="${PYTHONPATH}:/path/to/LLM-hallucination/"`

i.e. `export PYTHONPATH="${PYTHONPATH}:/home/genglin2/LLM-hallucination"`


- set up your own OpenAI API, Google BARD API, Replicate API
`source api_key_config.sh`


## datasets 

Tasks: `FalseQA`, `NEC`, `RefuNQ` (each task has two modes: `answerable` and `unanswerable`)

```python
data_path = f"/data/{task_name}/{task_name}_{mode}.json"  
```

Results are stored at `experiment_outputs/Logits`
Figures are saved at `experiment_outputs/figures`


## Setup for running the logit experiments on LLama-2 

 - install dependencies `pip install -r requirements.txt`
 - run `python src/run_llama_all.py`


## Quick Start

### generate LLM responses on FalseQA / NEC / RefuNQ

Our work contains three tasks:
 - FalseQA: answering questions that may contain false premises
 - Non-existent Concepts (NEC): explain questions that might involve nonexistent concepts
 - RefuNQ: refusal-inducing NaturalQuestions

To run different models on these tasks, we have 

`python src/run_*.py --prompt baseline` 

- For claude, you may need to run `ulimit -n 2048` to prevent a potential `too many open files` error. 


### Evaluation

To evaluate the outputs of the LLMs and visualize the analysis, see the notebooks in `/scripts`.


## Prompts
The prompts used in this repo can be found in the `prompts/` folder. 


## Citation

```
@misc{liu2023prudent,
      title={Prudent Silence or Foolish Babble? Examining Large Language Models' Responses to the Unknown}, 
      author={Genglin Liu and Xingyao Wang and Lifan Yuan and Yangyi Chen and Hao Peng},
      year={2023},
      eprint={2311.09731},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```