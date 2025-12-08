/*
  Allen Cabrera, Avanish Kulkarni
  This file will contain the driver to run the CUDA kernels on the device
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "load_tokens.h"

#define NUM_STEPS 100
#define N_LAYER 12

int *load_tokens(const char *filename, int *num_tokens);
GPT2Model *load_model(const char *filename);
void free_model(GPT2Model *model);
int inference(GPT2Model *model, int *tokens, int num_tokens, int k, int *top_indices, float *top_scores);
int main()
{
    // Initialize - start CUDA device, load model, allocate memory, create current state
    // Read arg1 for input
    const char *model_path = "./gpt2_124m.bin";
    const char *tokens_path = "./work/tokens.bin";
    const char *output_path = "./work/cuda_output.bin";
    int *tokens, num_tokens;
    GPT2Model *model = load_model(model_path);
    int k = 5;
    int top_indices[k];
    float top_scores[k];
    if (!model)
    {
        printf("Failed to load model\n");
        return 1;
    }

    // Input Processing - encode/tokenize input text, get tokens
    tokens = load_tokens(tokens_path, &num_tokens);
    if (!tokens)
    {
        printf("Failed to load tokens\n");
        free_model(model);
        return 1;
    }
    // call inference in cuda
    inference(model, tokens, num_tokens, k, top_indices, top_scores);

    printf("\nTop 5 predictions for the next token:\n");
    for (int i = 0; i < 5; i++)
    {
        printf("%d. Token ID: %d, Score: %.4f\n", i + 1, top_indices[i], top_scores[i]);
    }

    FILE *f = fopen(output_path, "wb");
    fwrite(top_indices, sizeof(int), k, f);
    fclose(f);

    printf("\nTop 5 token IDs saved to %s\n", output_path);

    // Free memory
    free(tokens);
    free_model(model);
    return 0;
}