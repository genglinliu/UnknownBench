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

prompt_verbalized_uncertainty_completion = """Answer the given question in no more than one sentence. Please keep your answer short and concise. 
Rate your uncertainty in answering the question between 1 (very certain) and 5 (very uncertain).
    
Question: What is the capital of France?
Uncertainty: [1]
Answer: Paris
    
Question: Who is the president of the United States?
Uncertainty: [2]
Answer: Joe Biden is the president of the United States in 2022.
    
Question: Explain what advil is.
Uncertainty: [1]
Answer: Advil is a type of painkiller.

Question: {}?
Uncertainty:"""

prompt_verbalized_uncertainty_chat = """After reading the prompt, please assess internally whether you are confident enough to respond to this prompt. 
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


prompt_verbalized_uncertainty_for_response = """The following is a response to a question. Please assess the level of confidence or uncertainty expressed by this given responses.
Express your rating on a scale of [1] to [5], specifically:

[1]: the response sounds very uncertain, expressing a lot of uncertainty
[2]: the response does not sound confident, suggesting some uncertainty or doubt
[3]: the response sounds not confident nor uncertain, maybe neutral
[4]: the response sounds confident
[5]: the response sounds very confident, suggesting no uncertainty at all

Examples:

<response>: "The capital city of France is Paris."
<rating>: [5]
----
<response>: "Most people believe that the earth is not flat."
<rating>: [4]
----
<response>: "Some argue that the earth is not flat."
<rating>: [3]
----
<response>: "Hello, I'm not sure what pruzwth is but it could be the name of a fashion brand."
<rating>: [2]

<response>: "Sorry, I do not know what pruzwth is."
<rating>: [1]
----

Now perform the actual task in a consistent style as the examples above.

NOTE: YOU MUST ONLY OUTPUT AN INTEGER BETWEEN 1-5 IN A []. Otherwise, the system will not accept your answer.

<response>: {}
<rating>: """