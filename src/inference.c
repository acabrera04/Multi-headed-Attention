/*
  Allen Cabrera, Avanish Kulkarni
  This file will contain the driver to run the CUDA kernels on the device
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include "model.h"
#include "../include/cuda_utils.h"
#include "load_tokens.h"

#define NUM_STEPS 100
#define N_LAYER 12

// Define model struct

// Define run state (buffers for activations)

// Load weights from file

// Initialize CUDA device

// Encode input data

// input1: model.bin, input2: tokens.bin
int main()
{
    // Initialize - start CUDA device, load model, allocate memory, create current state
    // Read arg1 for input
    const char *model_path = "../work/gpt2_124m.bin";
    const char *tokens_path = "../work/tokens.bin";
    int *tokens, num_tokens;
    GPT2Model *model = load_model(model_path);
    if (!model)
        return 1;

    // Input Processing - encode/tokenize input text, get tokens
    tokens = load_tokens(tokens_path, &num_tokens);
    if (!tokens)
        return 1;

    // call inference in cuda
    inference(model, tokens, num_tokens);

    // Free memory
    free_model(model);
    return 0;
}