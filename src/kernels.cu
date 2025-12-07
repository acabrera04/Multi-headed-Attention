// This file will contain CUDA kernels
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <unistd.h>
#include <sys/time.h>
#include "../include/cuda_utils.h"
#include "model.h"

#define MAX_GRID_DIM 1 << 12
#define MAX_BLOCK_DIM 1 << 10
#define BLOCK_DIM 16
#define THREADS 1 << 8
#define MAX_TILE_WIDTH 16

#define NUM_STEPS 100
#define N_LAYER 12

__global__ void mat_mult_cuda(float *d_a, float *d_b, float *d_c, int m, int n, int p, int tile_width)
{
    __shared__ int a_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ int b_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    int ph, k;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * tile_width + ty;
    int col = blockIdx.x * tile_width + tx;

    int sum = 0;
    for (ph = 0; ph < ceil(n / (float)tile_width); ph++)
    {
        int a_col = ph * tile_width + tx;
        int b_row = ph * tile_width + ty;
        if (row < m && a_col < n)
            a_shared[ty][tx] = d_a[row * n + a_col];
        else
            a_shared[ty][tx] = 0;

        if (col < p && b_row < n)
            b_shared[ty][tx] = d_b[b_row * p + col];
        else
            b_shared[ty][tx] = 0;

        __syncthreads();

        for (k = 0; k < tile_width; k++)
        {
            sum += a_shared[ty][k] * b_shared[k][tx];
        }
        __syncthreads();
    }
    if (row < m && col < p)
        d_c[row * p + col] = sum;
}

__global__ void mat_scalar_cuda(int *d_a, int m, int n, int scalar)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        d_a[row * m + col] *= scalar;
    }
}

__global__ void mat_add_cuda(float *d_a, float *d_b, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        d_a[row * m + col] += d_b[row * m + col];
    }
}

__global__ void softmax_cuda(int *A, int m, int n)
{
}

__global__ void token_embedding_cuda(int *tokens, float *output, int num_tokens, int embedding, float *embded_weights, float *pos_weights)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_tokens && col < embedding)
    {
        int token = tokens[row];
        output[row * embedding + col] = embded_weights[token * embedding + col] + pos_weights[row * embedding + col];
    }
}

__global__ void layer_norm_cuda(float *input, float *output, int num_tokens, float *weights, float *bias)
{
    int row = blockIdx.x;
    int tidx = threadIdx.x;

    __shared__ float smem[BLOCK_DIM * BLOCK_DIM];

    if (row < num_tokens)
    {
        float *in = input + row * N_EMBD;
        float *out = output + row * N_EMBD;
        // Calculate the mean for each row
        float local_mean = 0.0f;
        float local_variance = 0.0f;

        for (int i = tidx; i < N_EMBD; i += blockDim.x)
        {
            float a = in[i];
            local_mean += a;
            local_variance += a * a;
        }

        smem[tidx] = local_mean;
        __syncthreads();

        for (int stride = blockDim.x; stride > 0; stride /= 2)
        {
            if (tidx < stride)
            {
                smem[tidx] += smem[tidx + stride];
            }
            __syncthreads();
        }
        float mean = smem[0] / N_EMBD;
        __syncthreads();

        // Calculate the variance for each row
        smem[tidx] = local_variance;
        __syncthreads();

        for (int stride = blockDim.x; stride > 0; stride /= 2)
        {
            if (tidx < stride)
            {
                smem[tidx] += smem[tidx + stride];
            }
            __syncthreads();
        }
        float variance = (smem[0] / N_EMBD) - (mean * mean);
        float std_dev = sqrt(variance + 1e-5f);
        __syncthreads();
        // Normalize
        for (int i = tidx; i < N_EMBD; i += blockDim.x)
        {
            out[i] = weights[i] * (in[i] - mean) / std_dev + bias[i];
        }
    }
}

__global__ void gelu_cuda(float *x, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        float val = x[row * n + col];
        x[row * n + col] = 0.5f * val * (1.0f + tanh(sqrt(2.0f / M_PI) * (val + 0.044715f * pow(val, 3.0f))));
    }
}

int attention_cuda(int my_rank, int nprocs, int *h_Q, int *h_K, int *h_V, int *h_output, int q_rows, int q_cols, int k_rows, int k_cols, int v_rows, int v_cols)
{
    int i, iter;
    struct timeval timecheck;
    long dev_start, dev_end, dev_elapsed;

    gettimeofday(&timecheck, NULL);
    dev_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    int *d_Q, *d_K, *d_V, *d_output, *d_buffer;
    int buffer_rows = q_rows, buffer_cols = k_rows;
    unsigned int q_size = sizeof(int) * q_rows * q_cols, k_size = sizeof(int) * k_rows * k_cols, v_size = sizeof(int) * v_rows * v_cols;
    unsigned int output_size = sizeof(int) * q_rows * v_cols;
    cudaMalloc(reinterpret_cast<void **>(&d_Q), q_size);
    cudaMalloc(reinterpret_cast<void **>(&d_K), k_size);
    cudaMalloc(reinterpret_cast<void **>(&d_V), v_size);
    cudaMalloc(reinterpret_cast<void **>(&d_buffer), sizeof(int) * buffer_rows * buffer_cols);
    // TODO Confirm that this is correct size
    cudaMalloc(reinterpret_cast<void **>(&d_output), output_size);

    cudaMemcpy(d_Q, h_Q, q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, v_size, cudaMemcpyHostToDevice);

    int bx_dim, by_dim, gx_dim, gy_dim;

    by_dim = bx_dim;
    // TODO Figure out grid calculations and block dimensions
    gx_dim = ceil();
    gy_dim = ceil();

    dim3 grid(gx_dim, gy_dim);
    dim3 threads(bx_dim, by_dim);

    // TODO Figure out a tilewidth (replace 4), ensure that K is transposed before call
    // Handles Q*K^T
    mat_mult_cuda<<<grid, threads>>>(d_Q, d_K, d_buffer, q_rows, q_cols, k_rows, 4);

    // Multiples previous result by a scalar of the dimensions of k
    mat_scalar_cuda<<<grid, threads>>>(d_buffer, buffer_rows, buffer_cols, 1 / sqrt(q_cols));

    // Handles softmax of result of previous matrix multiplication call
    softmax_cuda<<<grid, threads>>>(d_buffer, buffer_rows, buffer_cols);

    // multiply the result of the softmax by V
    mat_mult_cuda<<<grid, threads>>>(d_buffer, d_V, d_output, buffer_rows, buffer_cols, v_cols, 4);

    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    gettimeofday(&timecheck, NULL);
    dev_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
    dev_elapsed = dev_end - dev_start;

    printf("dev time: rank=%d: %d procs: %ld msecs\n",
           my_rank, nprocs, dev_elapsed);
    fflush(stdout);

    cudaFree(d_Q);
    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_buffer);
    cudaFree(d_output);

    return 1;
}

// Adds token empeddings and position embedding. Creates a num_tokens x N_EMBD matrix
float *input_embedding(int *tokens, float *output, int num_tokens, float *embed_weights, float *pos_weights)
{
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N_EMBD + 15) / 16, ((num_tokens) + 15) / 16);
    token_embedding_cuda<<<grid, block>>>(tokens, output, num_tokens, N_EMBD, embed_weights, pos_weights);
}

void layer_norm(float *x, int num_tokens, LayerNorm *layerNorm, float *output)
{
    layer_norm_cuda<<<num_tokens, THREADS>>>(x, output, num_tokens, layerNorm->weight, layerNorm->bias);
}

void residual_connection(float *x, float *y, int num_tokens)
{
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, ((num_tokens) + BLOCK_DIM - 1) / BLOCK_DIM);
    mat_add_cuda<<<grid, block>>>(x, y, num_tokens, N_EMBD);
}

void attention(float *h, int m, int n, TransformerBlock *transformer, float *output) {}

void feed_forward(float *x, int num_tokens, TransformerBlock *transformer, float *output)
{
    // x: [num_tokens, N_EMBD]
    // c_fc: [N_EMBD, 4*N_EMBD]

    // hidden layer: [num_tokens, 4*N_EMBD]
    float *h;
    cudaMalloc(&h, sizeof(float) * num_tokens * 4 * N_EMBD);
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((4 * N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, ((num_tokens) + BLOCK_DIM - 1) / BLOCK_DIM);
    mat_mult_cuda<<<grid, block>>>(x, transformer->c_fc.weight, h, num_tokens, N_EMBD, 4 * N_EMBD, MAX_TILE_WIDTH);

    // add bias
    mat_add_cuda<<<grid, block>>>(h, transformer->c_fc.bias, num_tokens, 4 * N_EMBD);

    // gelu
    gelu_cuda<<<grid, block>>>(h, num_tokens, 4 * N_EMBD);

    // c_proj: [4*N_EMBD, N_EMBD]
    // now project based on model weights
    mat_mult_cuda<<<grid, block>>>(h, transformer->c_proj_mlp.weight, output, num_tokens, 4 * N_EMBD, N_EMBD, MAX_TILE_WIDTH);

    // add bias
    mat_add_cuda<<<grid, block>>>(output, transformer->c_proj_mlp.bias, num_tokens, N_EMBD);
    cudaFree(h);
}

void top_k(float *logits, int vocab_size, int k, int *top_indices, float *top_scores) {}

void logits(float *x, float *logits, int vocab_size, int k, int *top_indices, float *top_scores) {}

int inference(GPT2Model *model, int *tokens, int num_tokens)
{
    float *token_embeddings;
    float *weight, *bias, *h, *attn_out, *mlp_out, *logits_arr, *token_embeddings;
    TransformerBlock *transformer;
    int *Q, *K, *V;
    int top_indices[5];
    float top_scores[5];
    cudaError_t cudaError;
    cudaError = cudaMalloc(&token_embeddings, sizeof(float) * num_tokens * N_EMBD);

    cudaError = cudaMalloc(&h, sizeof(float) * num_tokens * N_EMBD);

    cudaError = cudaMalloc(&attn_out, sizeof(float) * num_tokens * N_EMBD);

    cudaError = cudaMalloc(&mlp_out, sizeof(float) * num_tokens * N_EMBD);

    cudaError = cudaMalloc(&logits_arr, sizeof(float) * VOCAB_SIZE);

    for (int step = 0; step < NUM_STEPS; ++step)
    {
        // Launch CUDA kernel for inference

        // Lookup embedding
        input_embedding(tokens, token_embeddings, num_tokens, model->wte, model->wpe);

        // Transformer Layers
        for (int l = 0; l < N_LAYER; l++)
        {
            transformer = &(model->h[l]);
            // Layer Norm 1
            // returns a numOfTokens x N_EMBD (768) matrix
            layer_norm(token_embeddings, num_tokens, &transformer->ln_1, h);
            // Multi-Head Attention
            //    a. QKV Projection
            //    b. Attention Mechanism (Softmax(Q*K^T / sqrt(d_k)) * V)
            //    c. Output Projection

            attention(h, num_tokens, N_EMBD, transformer, attn_out);
            // attention_cuda();

            // Residual Connection 1
            residual_connection(token_embeddings, attn_out, num_tokens);

            // Layer Norm 2
            layer_norm(token_embeddings, num_tokens, &transformer->ln_2, h);
            // Feed Forward Network (GELU activation)
            //    a. Linear -> GELU
            //    b. Linear
            feed_forward(h, num_tokens, transformer, mlp_out);

            // Residual Connection 2
            residual_connection(token_embeddings, mlp_out, num_tokens);
        }

        // Final Layer Norm
        layer_norm(token_embeddings, num_tokens, N_EMBD, &model->ln_f, h);

        // Logits (MatMul with embedding table)
        logits(h, logits_arr, VOCAB_SIZE, 5, top_indices, top_scores);
        // Synchronize to ensure kernel completion
        cudaDeviceSynchronize();
        printf("\nTop 5 predictions for the next token in step %d:\n", step);
        for (int i = 0; i < 5; i++)
        {
            printf("%d. Token ID: %d, Score: %.4f\n", i + 1, top_indices[i], top_scores[i]);
        }
        // Next Token Selection
        // printf("%s", decode_token(current_token));
    }

    cudaFree(h);
    cudaFree(attn_out);
    cudaFree(mlp_out);
    cudaFree(logits_arr);
    cudaFree(token_embeddings);
}