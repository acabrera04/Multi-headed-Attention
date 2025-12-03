#include <stdio.h>
#include <stdlib.h>
#include "model.h"

// helper to allocate memory and copy data from file
float* load_tensor(FILE *f, size_t size) {
  float *data = (float*)malloc(size * sizeof(float));
  if (!data) {
    fprintf(stderr, "failed to allocate memory\n");
    exit(1);
  }

  if (fread(data, sizeof(float), size, f) != size) {
    fprintf(stderr, "failed to read tensor data\n");
    free(data);
    exit(1);
  }

  return data;
}

// load model from file serialized by serialize_model.py, mirroring the hardcoded layout
GPT2Model* load_model(const char* filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    printf("failed to open model file: %s\n", filename);
    return NULL;
  }

  GPT2Model *model = (GPT2Model*)malloc(sizeof(GPT2Model));

  model->wte = load_tensor(f, VOCAB_SIZE * N_EMBD);
  model->wpe = load_tensor(f, MAX_POS_EMBD * N_EMBD);

  for (int i = 0; i < N_LAYER; i++) {
    // layer norm 1
    model->h[i].ln_1.weight = load_tensor(f, N_EMBD);
    model->h[i].ln_1.bias = load_tensor(f, N_EMBD);

    // attention weights
    model->h[i].c_attn.weight = load_tensor(f, N_EMBD * 3 * N_EMBD);
    model->h[i].c_attn.bias = load_tensor(f, 3 * N_EMBD);
    model->h[i].c_proj.weight = load_tensor(f, N_EMBD * N_EMBD);
    model->h[i].c_proj.bias = load_tensor(f, N_EMBD);

    // layer norm 2
    model->h[i].ln_2.weight = load_tensor(f, N_EMBD);
    model->h[i].ln_2.bias = load_tensor(f, N_EMBD);

    // mlp weights
    model->h[i].c_fc.weight = load_tensor(f, N_EMBD * 4 * N_EMBD);
    model->h[i].c_fc.bias = load_tensor(f, 4 * N_EMBD);
    model->h[i].c_proj_mlp.weight = load_tensor(f, 4 * N_EMBD * N_EMBD);
    model->h[i].c_proj_mlp.bias = load_tensor(f, N_EMBD);
  }

  // 3. Final Layer Norm
  model->ln_f.weight = load_tensor(f, N_EMBD);
  model->ln_f.bias = load_tensor(f, N_EMBD);

  fclose(f);
  printf("model loaded to memory.\n");
  return model;
}

void free_model(GPT2Model* model) {
  if (!model) return;

  free(model->wte);
  free(model->wpe);

  for (int i = 0; i < N_LAYER; i++) {
    free(model->h[i].ln_1.weight);
    free(model->h[i].ln_1.bias);
    free(model->h[i].c_attn.weight);
    free(model->h[i].c_attn.bias);
    free(model->h[i].c_proj.weight);
    free(model->h[i].c_proj.bias);
    free(model->h[i].ln_2.weight);
    free(model->h[i].ln_2.bias);
    free(model->h[i].c_fc.weight);
    free(model->h[i].c_fc.bias);
    free(model->h[i].c_proj_mlp.weight);
    free(model->h[i].c_proj_mlp.bias);
  }

  free(model->ln_f.weight);
  free(model->ln_f.bias);

  free(model);
}
