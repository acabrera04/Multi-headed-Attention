import torch
from transformers import GPT2Model
import numpy as np

model = GPT2Model.from_pretrained("gpt2")
model.eval()

def write_tensor(f, tensor):
  arr = tensor.detach().cpu().numpy().astype(np.float32)
  f.write(arr.tobytes())

with open("../work/gpt2_124m.bin", "wb") as f:
  # embeddings
  write_tensor(f, model.wte.weight)  # token embeddings
  write_tensor(f, model.wpe.weight)  # positional embeddings

  # transformer blocks
  for block in model.h:
    # layer norm 1
    write_tensor(f, block.ln_1.weight)
    write_tensor(f, block.ln_1.bias)

    # attention weights
    write_tensor(f, block.attn.c_attn.weight)
    write_tensor(f, block.attn.c_attn.bias)
    write_tensor(f, block.attn.c_proj.weight)
    write_tensor(f, block.attn.c_proj.bias)

    # layer norm 2
    write_tensor(f, block.ln_2.weight)
    write_tensor(f, block.ln_2.bias)

    # mlp weights
    write_tensor(f, block.mlp.c_fc.weight)
    write_tensor(f, block.mlp.c_fc.bias)
    write_tensor(f, block.mlp.c_proj.weight)
    write_tensor(f, block.mlp.c_proj.bias)

  # final layer norm
  write_tensor(f, model.ln_f.weight)
  write_tensor(f, model.ln_f.bias)