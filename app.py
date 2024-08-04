from transformers import pipeline
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = "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"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
    token=access_token
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipe(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
assistant_response = outputs[0]["generated_text"][-1]["content"]
print(assistant_response)
