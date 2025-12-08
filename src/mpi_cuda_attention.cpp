#ifndef M_PI
#define M_PI 3.1415926535897932384626433832
#endif

#include "load_tokens.h"
#include "model.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"

int *load_tokens(const char *filename, int *num_tokens);
GPT2Model *load_model(const char *filename);
void free_model(GPT2Model *model);
int inference(GPT2Model *model, int *tokens, int num_tokens, int k, int *top_indices, float *top_scores);
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > N_HEAD)
    {
        if (rank == 0)
            printf("Warning: Number of ranks is greater than N_HEAD (12) and some ranks will not be used.\n");
    }
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
    if (rank == 0) {
        tokens = load_tokens(tokens_path, &num_tokens);
        printf("rank 0 has loaded %d tokens\n", num_tokens);
        if (!tokens) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&num_tokens, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Rank %d: numtokens: %d\n", rank, num_tokens);

    if (rank != 0) {
        tokens = (int *)malloc(num_tokens * sizeof(int));
    }

    MPI_Bcast(tokens, num_tokens, MPI_INT, 0, MPI_COMM_WORLD);
    
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
    MPI_Finalize();

    return 0;
}