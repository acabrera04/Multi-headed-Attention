import sys
import struct
from transformers import GPT2Tokenizer


def load_tokens(filename):
    with open(filename, "rb") as f:
        data = f.read()
        tokens = struct.unpack("{}i".format(len(data) // 4), data)
    return tokens


def main():
    if len(sys.argv) < 2:
        print("Usage: python tokens_to_text.py <tokens.bin>")
        sys.exit(1)

    input_file = sys.argv[1]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = load_tokens(input_file)
    text = tokenizer.decode(tokens)

    print(f"Decoded '{tokens}' to {text}.")


if __name__ == "__main__":
    main()
