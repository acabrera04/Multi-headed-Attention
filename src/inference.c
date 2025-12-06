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
int main(int argc, char **argv)
{
    // Initialize - start CUDA device, load model, allocate memory, create current state
    // Read arg1 for input
    char *modelFileName, *tokenFileName;
    int *tokens, numOfTokens;
    GPT2Model *model = load_model(modelFileName);

    // Input Processing - encode/tokenize input text, get tokens
    tokens = load_tokens(tokenFileName, &numOfTokens);

    // call inference in cuda
    inference(model, tokens, numOfTokens);

    // Free memory
    free_model(model);
    return 0;
}