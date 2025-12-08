import sys
import struct
from transformers import GPT2Tokenizer

if len(sys.argv) < 2:
    print("Usage: python tokenize_to_bin.py <text>")
    sys.exit(1)

text = sys.argv[1]
tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer/")
tokens = tokenizer.encode(text, add_special_tokens=False)

output_file = "./work/tokens.bin"
with open(output_file, "wb") as f:
    for token in tokens:
        f.write(struct.pack("i", token))

print(f"Tokenized '{text}' to {tokens}.")
print(f"Saved to {output_file}")

