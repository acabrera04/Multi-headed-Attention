import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

# select CUDA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on {device}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def next_tokens(input_text, k):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # get next possible token
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    next_token_logits = logits[:, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)  # and their probabilities

    # get top k tokens
    top_k_probs, top_k_ids = torch.topk(probs, k)
    top_k_tokens = [tokenizer.decode([tid]) for tid in top_k_ids[0]]

    return list(zip(top_k_tokens, top_k_probs[0].tolist()))

input_text = input("Enter text: ")
start = time.time()
predictions = next_tokens(input_text, 5)
end = time.time()
print("Next token candidates:")
for n, (token, prob) in enumerate(predictions):
    print(f"{n+1}. {repr(token)} (prob: {prob})")
print(f'\n Reference Inference time: {(end-start) * 1000:.2f} ms')