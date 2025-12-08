import sys
import os
import struct
from transformers import GPT2Tokenizer


def load_tokens(filename):
    with open(filename, "rb") as f:
        data = f.read()
        tokens = struct.unpack("{}i".format(len(data) // 4), data)
    return tokens


def main():
    input_file = None
    if len(sys.argv) != 2:
        print("reading from default file '../work/output.bin'")
        input_file = (r'../work/output.bin')
    else:
        input_file = sys.argv[1]
    
    tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer/")
    tokens = list(load_tokens(input_file))
    print(f"Decoded {len(tokens)} tokens from '{input_file}' to text:")

    for i, t in enumerate(tokens):
        t = tokenizer.decode([t])
        print(f'{i+1}. {t}')


if __name__ == "__main__":
    main()
