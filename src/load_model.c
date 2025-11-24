#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "model.h"

// helper to allocate device memory and copy data from file, returns device pointer
float* load_tensor(FILE *f, size_t size) {
  float *h_data = (float*)malloc(size * sizeof(float));
  if (!h_data) {
    fprintf(stderr, "failed to allocate host memory\n");
    exit(1);
  }

  if (fread(h_data, sizeof(float), size, f) != size) {
    fprintf(stderr, "failed to read tensor data\n");
    free(h_data);
    exit(1);
  }

  float *d_data;
  checkCudaError(cudaMalloc((void**)&d_data, size * sizeof(float)));

  checkCudaError(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));

  free(h_data);
  return d_data;
}

// load model from file serialized by serialize_model.py, mirroring the hardcoded layout
GPT2Model* load_model(const char* filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    printf("failed to open model file: %s\n", filename);
    return;
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
  printf("model loaded to device memory.\n");
  return model;
}

void free_model(GPT2Model* model) {
  if (!model) return;

  cudaFree(model->wte);
  cudaFree(model->wpe);

  for (int i = 0; i < N_LAYER; i++) {
    cudaFree(model->h[i].ln_1.weight);
    cudaFree(model->h[i].ln_1.bias);
    cudaFree(model->h[i].c_attn.weight);
    cudaFree(model->h[i].c_attn.bias);
    cudaFree(model->h[i].c_proj.weight);
    cudaFree(model->h[i].c_proj.bias);
    cudaFree(model->h[i].ln_2.weight);
    cudaFree(model->h[i].ln_2.bias);
    cudaFree(model->h[i].c_fc.weight);
    cudaFree(model->h[i].c_fc.bias);
    cudaFree(model->h[i].c_proj_mlp.weight);
    cudaFree(model->h[i].c_proj_mlp.bias);
  }

  cudaFree(model->ln_f.weight);
  cudaFree(model->ln_f.bias);

  free(model);
}
