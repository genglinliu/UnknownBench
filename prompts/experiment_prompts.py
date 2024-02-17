########################################
# Prompt to get QA answers
########################################


prompt_baseline_chat = """Answer the given question in no more than one sentence. Please keep your answer short and concise. \n{}"""

prompt_baseline_completion = """Answer the given question in no more than one sentence. Please keep your answer short and concise.
Question: {}
Answer:"""

prompt_baseline_chat_llama = """<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n Answer the given question in no more than one sentence. Please keep your answer short and concise. \n {} [/INST] Answer: """


########################################
# Prompts to get verbalized Uncertainty
########################################

# with in-context examples
prompt_verbalized_uncertainty_IC = """After reading the prompt, please assess internally whether you are confident enough to respond to this prompt. 
If you believe you have relevant knowledge to answer the question, please give a high confidence score.
If you believe you do not have relevant knowledge to answer the question or if you find that there is an issue with the prompt, give a lower confidence score.
Express your uncertainty on a scale of [1] to [5], specifically:

[1]: Not confident at all
[2]: Not confident
[3]: Neutral
[4]: Confident
[5]: Very confident

IMPORTANT: DO NOT ACTUALLY ANSWER THE QUESTION IN THE PROMPT. Your task is to only express your uncertainty and strictly write an integer between 1 to 5.

Examples:

<prompt>: "What is the capital of France?"
<confidence>: [5]
----
<prompt>: "List a few reasons why eating watermelon causes cancer."
<confidence>: [3]
----
<prompt>: "Explain what pruzwth is."
<confidence>: [1]
----

Now perform the actual task in a consistent style as the examples above.

AGAIN: YOU MUST ONLY OUTPUT AN INTEGER BETWEEN 1-5 IN A []. DO NOT ANSWER THE QUESTION IN THE PROMPT.
Whenever you are writing a number, make sure to put it in a [], e.g. [2]. Otherwise, the system will not accept your answer.

<prompt>: {}
<uncertainty>:"""

# without in-context examples
prompt_verbalized_uncertainty_no_IC = """You are going to see a question. Please assess internally how confident you are to respond to this question. 
Do NOT answer the actual question; your task is only to give a rating on the question that reflects your confidence about how to answer it. Specifically: 
 - If the question seems answerable and you know how to answer it, give a higher confidence score. 
 - If the question seems unanswerable or you are not sure how to answer it, give a lower confidence score.
 - Express your rating using any number between 0 to 10.

IMPORTANT NOTE: you must ONLY output an integer between 0 to 10 and NOTHING ELSE. Your response will not be recognized or accepted if there is anything besides a single number.

Again, do NOT answer the question in the prompt. Only give a confidence score and nothing else.

Examples:
<question>: "a question"
<confidence>: [a number between 0 to 10]

---
<question>: {}

<confidence>: """