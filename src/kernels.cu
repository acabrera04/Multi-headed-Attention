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

#define CHECK_CUDA_KERNEL(name)                                               \
    do                                                                        \
    {                                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess)                                               \
        {                                                                     \
            fprintf(stderr, "CUDA kernel %s failed at %s:%d: %s\n",           \
                    name, __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
        cudaDeviceSynchronize();                                              \
        err = cudaGetLastError();                                             \
        if (err != cudaSuccess)                                               \
        {                                                                     \
            fprintf(stderr, "CUDA kernel %s execution failed at %s:%d: %s\n", \
                    name, __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)
__global__ void mat_mult_cuda(float *d_a, float *d_b, float *d_c, int m, int n, int p, int tile_width)
{
    __shared__ float a_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ float b_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    int ph, k;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * tile_width + ty;
    int col = blockIdx.x * tile_width + tx;

    float sum = 0;
    for (ph = 0; ph < (n + tile_width - 1) / tile_width; ph++)
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

__global__ void mat_add_cuda(float *d_a, float *d_b, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        d_a[row * n + col] += d_b[row * n + col];
    }
}

__global__ void mat_add_bias_cuda(float *mat, float *bias, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        mat[row * n + col] += bias[col];
    }
}

__global__ void softmax_cuda(float *A, int m, int n)
{
    __shared__ float smem[BLOCK_DIM * BLOCK_DIM];
    int row = blockIdx.x;
    int tidx = threadIdx.x;

    if (row < m)
    {
        float *row_ptr = A + row * n;
        float local_max = -INFINITY;
        for (int i = tidx; i < n; i += blockDim.x)
        {
            float x = row_ptr[i];
            if (x > local_max)
            {
                local_max = x;
            }
        }
        smem[tidx] = local_max;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (tidx < stride)
            {
                smem[tidx] = fmaxf(smem[tidx], smem[tidx + stride]);
            }
            __syncthreads();
        }
        float max_val = smem[0];
        __syncthreads();
        float sum = 0.0f;
        for (int i = tidx; i < n; i += blockDim.x)
        {
            float x = row_ptr[i];
            sum += expf(x - max_val);
        }
        smem[tidx] = sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (tidx < stride)
            {
                smem[tidx] += smem[tidx + stride];
            }
            __syncthreads();
        }
        float norm = smem[0];
        __syncthreads();

        for (int i = tidx; i < n; i += blockDim.x)
        {
            row_ptr[i] = expf(row_ptr[i] - max_val) / norm;
        }
    }
}

__global__ void token_embedding_cuda(int *tokens, float *output, int num_tokens, float *embded_weights, float *pos_weights)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_tokens && col < N_EMBD)
    {
        int token = tokens[row];
        output[row * N_EMBD + col] = embded_weights[token * N_EMBD + col] + pos_weights[row * N_EMBD + col];
    }
}

__global__ void layer_norm_cuda(float *input, float *output, int num_tokens, float *weights, float *bias)
{
    int row = blockIdx.x;
    int tidx = threadIdx.x;

    __shared__ float smem[THREADS];

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

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
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

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tidx < stride)
            {
                smem[tidx] += smem[tidx + stride];
            }
            __syncthreads();
        }
        float variance = (smem[0] / N_EMBD) - (mean * mean);
        float inv_std = rsqrt(variance + 1e-5f);
        __syncthreads();
        // Normalize
        for (int i = tidx; i < N_EMBD; i += blockDim.x)
        {
            out[i] = weights[i] * (in[i] - mean) * inv_std + bias[i];
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

__global__ void qkv_decompose_cuda(float *qkv, float *q, float *k, float *v, int num_tokens, int n_head, int head_dim)
{
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    int feat = blockIdx.y * blockDim.y + threadIdx.y;
    int head = blockIdx.z * blockDim.z + threadIdx.z;

    if (token < num_tokens && head < n_head && feat < head_dim)
    {
        int idx = num_tokens * head_dim * head + head_dim * token + feat;

        int qkv_offset = token * 3 * N_EMBD;

        // Q is first N_EMBD, K is second N_EMBD, V is third N_EMBD (so offset by N_EMBD and 2*N_EMBD)
        q[idx] = qkv[qkv_offset + head * head_dim + feat];
        k[idx] = qkv[qkv_offset + N_EMBD + head * head_dim + feat];
        v[idx] = qkv[qkv_offset + 2 * N_EMBD + head * head_dim + feat];
    }
}

__global__ void scaled_dot_product_cuda(float *q, float *k, float *att, float scale, int num_tokens, int head_dim)
{
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    int token2 = blockIdx.y * blockDim.y + threadIdx.y;
    int head = blockIdx.z * blockDim.z + threadIdx.z;

    if (token < num_tokens && token2 < num_tokens && head < N_HEAD)
    {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; i++)
        {
            sum += q[head * num_tokens * head_dim + token * head_dim + i] *
                   k[head * num_tokens * head_dim + token2 * head_dim + i];
        }
        att[head * num_tokens * num_tokens + token * num_tokens + token2] = sum * scale;
    }
}

__global__ void mask_cuda(float *att, int num_tokens, int n_head)
{
    int token2 = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y * blockDim.y + threadIdx.y;

    if (token < num_tokens && token2 < num_tokens && token2 > token)
    {
        for (int head = 0; head < n_head; head++)
        {
            att[head * num_tokens * num_tokens + token * num_tokens + token2] = -1e10f;
        }
    }
}

__global__ void compute_logits_cuda(float *logits, float *last_token_emb, float *wte)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < VOCAB_SIZE)
    {
        float sum = 0.0f;
        for (int j = 0; j < N_EMBD; j++)
        {
            sum += last_token_emb[j] * wte[i * N_EMBD + j];
        }
        logits[i] = sum;
    }
}

void checkCuda(const char *func, const char *state)
{
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("%s kernel %s failed: %s\n", func, state, cudaGetErrorString(err));
    }
    checkCudaErrors(err);
}

// Adds token empeddings and position embedding. Creates a num_tokens x N_EMBD matrix
void input_embedding(int *tokens, float *output, int num_tokens, float *embed_weights, float *pos_weights)
{
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, (num_tokens + BLOCK_DIM - 1) / BLOCK_DIM);
    token_embedding_cuda<<<grid, block>>>(tokens, output, num_tokens, embed_weights, pos_weights);
    CHECK_CUDA_KERNEL("input embedding");
}

void layer_norm(float *x, int num_tokens, LayerNorm *layerNorm, float *output)
{
    layer_norm_cuda<<<num_tokens, THREADS>>>(x, output, num_tokens, layerNorm->weight, layerNorm->bias);
    CHECK_CUDA_KERNEL("layer norm");
}

void residual_connection(float *x, float *y, int num_tokens)
{
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, ((num_tokens) + BLOCK_DIM - 1) / BLOCK_DIM);
    mat_add_cuda<<<grid, block>>>(x, y, num_tokens, N_EMBD);
    CHECK_CUDA_KERNEL("residual mat add");
}

void attention(float *x, int num_tokens, TransformerBlock *transformer, float *output)
{
    int head_dim = N_EMBD / N_HEAD;

    float *qkv;
    checkCudaErrors(cudaMalloc(&qkv, sizeof(float) * num_tokens * 3 * N_EMBD));

    dim3 block1(BLOCK_DIM, BLOCK_DIM);
    dim3 grid1((3 * N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, ((num_tokens) + BLOCK_DIM - 1) / BLOCK_DIM);

    // project qkv
    mat_mult_cuda<<<grid1, block1>>>(x, transformer->c_attn.weight, qkv, num_tokens, N_EMBD, 3 * N_EMBD, MAX_TILE_WIDTH);
    CHECK_CUDA_KERNEL("attention mat_mult1");

    // add bias
    mat_add_bias_cuda<<<grid1, block1>>>(qkv, transformer->c_attn.bias, num_tokens, 3 * N_EMBD);
    CHECK_CUDA_KERNEL("attention add_bias1");

    // decompose qkv matrix into Q, K, V
    float *q, *k, *v;
    checkCudaErrors(cudaMalloc(&q, sizeof(float) * num_tokens * N_HEAD * head_dim));
    checkCudaErrors(cudaMalloc(&k, sizeof(float) * num_tokens * N_HEAD * head_dim));
    checkCudaErrors(cudaMalloc(&v, sizeof(float) * num_tokens * N_HEAD * head_dim));

    // hard coded for now since this is a special 3d case
    dim3 block2(8, 8, 4);
    dim3 grid2((num_tokens + 7) / 8, (head_dim + 7) / 8, (N_HEAD + 3) / 4);
    qkv_decompose_cuda<<<grid2, block2>>>(qkv, q, k, v, num_tokens, N_HEAD, head_dim);

    CHECK_CUDA_KERNEL("attention qkv decompose");

    cudaFree(qkv);

    // now computing attention scores = dot product between q and k.T
    // q: [N_HEAD, num_tokens, head_dim]
    // k: [N_HEAD, num_tokens, head_dim]
    // leads to matrix of size: [N_HEAD, num_tokens, num_tokens], the attention matrix
    float *att;
    checkCudaErrors(cudaMalloc(&att, N_HEAD * num_tokens * num_tokens * sizeof(float)));

    float scale = 1.0f / sqrt(head_dim);
    dim3 grid3((num_tokens + 7) / 8, (num_tokens + 7) / 8, (N_HEAD + 3) / 4);

    scaled_dot_product_cuda<<<grid3, block2>>>(q, k, att, scale, num_tokens, head_dim);

    CHECK_CUDA_KERNEL("attention scaled dot_product1");

    cudaFree(q);
    cudaFree(k);

    // causal mask (a lower triangular matrix)
    // so that the model cannot attend to future tokens
    // basically this means setting upper triangular values to -inf (we use a large negative number)
    // whcih will be zeroed out in softmax
    dim3 grid4((num_tokens + BLOCK_DIM - 1) / BLOCK_DIM, (num_tokens + BLOCK_DIM - 1) / BLOCK_DIM);
    mask_cuda<<<grid4, block1>>>(att, num_tokens, N_HEAD);

    CHECK_CUDA_KERNEL("attention mask");

    // softmax
    // since its a 3D matrix flattened to 1D, we can just treat it as a 2D matrix of size:
    // [N_HEAD * num_tokens, num_tokens] to apply softmax row-wise
    softmax_cuda<<<N_HEAD * num_tokens, THREADS>>>(att, N_HEAD * num_tokens, num_tokens);

    CHECK_CUDA_KERNEL("attention softmax");

    // now compute output values for each head with the attention weights and dot product with v
    // reminder: v is from qkv matrix
    // att: [N_HEAD, num_tokens, num_tokens]
    // v: [N_HEAD, num_tokens, head_dim]
    // y: [N_HEAD, num_tokens, head_dim]

    float *y_heads;
    checkCudaErrors(cudaMalloc(&y_heads, N_HEAD * num_tokens * head_dim * sizeof(float)));

    // THIS IS NOT A RIGHT TRANSLATION
    scaled_dot_product_cuda<<<grid2, block2>>>(att, v, y_heads, 1, num_tokens, head_dim);

    CHECK_CUDA_KERNEL("attention dot_product2");

    cudaFree(att);
    cudaFree(v);

    float *y_reassembled;
    checkCudaErrors(cudaMalloc(&y_reassembled, num_tokens * N_HEAD * sizeof(float)));
    // idk some code to assign heads to y
    cudaFree(y_heads);

    mat_mult_cuda<<<grid1, block1>>>(y_reassembled, transformer->c_proj.weight, output, num_tokens, N_EMBD, N_EMBD, MAX_TILE_WIDTH);
    CHECK_CUDA_KERNEL("attention mat_mult2");

    mat_add_bias_cuda<<<grid1, block1>>>(output, transformer->c_proj.bias, num_tokens, N_EMBD);

    CHECK_CUDA_KERNEL("attention add_bias2");

    cudaFree(y_reassembled);
}

void feed_forward(float *x, int num_tokens, TransformerBlock *transformer, float *output)
{
    // x: [num_tokens, N_EMBD]
    // c_fc: [N_EMBD, 4*N_EMBD]

    // hidden layer: [num_tokens, 4*N_EMBD]
    float *h;
    checkCudaErrors(cudaMalloc(&h, sizeof(float) * num_tokens * 4 * N_EMBD));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((4 * N_EMBD + BLOCK_DIM - 1) / BLOCK_DIM, ((num_tokens) + BLOCK_DIM - 1) / BLOCK_DIM);
    mat_mult_cuda<<<grid, block>>>(x, transformer->c_fc.weight, h, num_tokens, N_EMBD, 4 * N_EMBD, MAX_TILE_WIDTH);

    CHECK_CUDA_KERNEL("feed forward mat_mult1");

    // add bias
    mat_add_bias_cuda<<<grid, block>>>(h, transformer->c_fc.bias, num_tokens, 4 * N_EMBD);

    CHECK_CUDA_KERNEL("feed forward add_bias1");

    // gelu
    gelu_cuda<<<grid, block>>>(h, num_tokens, 4 * N_EMBD);

    CHECK_CUDA_KERNEL("feed forward gelu");

    // c_proj: [num_tokens, N_EMBD]
    // now project based on model weights
    mat_mult_cuda<<<grid, block>>>(h, transformer->c_proj_mlp.weight, output, num_tokens, 4 * N_EMBD, N_EMBD, MAX_TILE_WIDTH);

    CHECK_CUDA_KERNEL("feed forward mat_mult2");

    // add bias
    mat_add_bias_cuda<<<grid, block>>>(output, transformer->c_proj_mlp.bias, num_tokens, N_EMBD);

    CHECK_CUDA_KERNEL("feed forward add_bias2");

    cudaFree(h);
}

// not sure if worth it to parrallize
void top_k(float *logits, int vocab_size, int k, int *top_indices, float *top_scores)
{
    for (int i = 0; i < k; i++)
    {
        top_scores[i] = -1e9f;
    }

    for (int i = 0; i < vocab_size; i++)
    {
        float score = logits[i];
        for (int j = 0; j < k; j++)
        {
            if (score > top_scores[j])
            {
                for (int m = k - 1; m > j; m--)
                {
                    top_scores[m] = top_scores[m - 1];
                    top_indices[m] = top_indices[m - 1];
                }
                top_scores[j] = score;
                top_indices[j] = i;
                break;
            }
        }
    }
}

void logits(float *x, float *logits, float *wte)
{
    compute_logits_cuda<<<(VOCAB_SIZE + THREADS - 1) / THREADS, THREADS>>>(logits, x, wte);
    CHECK_CUDA_KERNEL("logits");
}

int inference(GPT2Model *model, int *tokens, int num_tokens)
{
    float *h, *attn_out, *mlp_out, *d_logits, *token_embeddings, h_logits[VOCAB_SIZE];
    int *d_tokens;
    TransformerBlock *transformer;
    int top_indices[5];
    float top_scores[5];

    checkCudaErrors(cudaMalloc(&d_tokens, sizeof(int) * num_tokens));
    checkCudaErrors(cudaMemcpy(d_tokens, tokens, sizeof(int) * num_tokens, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&token_embeddings, sizeof(float) * num_tokens * N_EMBD));

    checkCudaErrors(cudaMalloc(&h, sizeof(float) * num_tokens * N_EMBD));

    checkCudaErrors(cudaMalloc(&attn_out, sizeof(float) * num_tokens * N_EMBD));

    checkCudaErrors(cudaMalloc(&mlp_out, sizeof(float) * num_tokens * N_EMBD));

    checkCudaErrors(cudaMalloc(&d_logits, sizeof(float) * VOCAB_SIZE));

    for (int step = 0; step < 1; ++step)
    {
        // Launch CUDA kernel for inference

        // Lookup embedding
        input_embedding(d_tokens, token_embeddings, num_tokens, model->wte, model->wpe);
        printf("Input embedding success!\n\n");
        // Transformer Layers
        for (int l = 0; l < N_LAYER; l++)
        {
            transformer = &(model->h[l]);
            // Layer Norm 1
            // returns a numOfTokens x N_EMBD (768) matrix
            layer_norm(token_embeddings, num_tokens, &transformer->ln_1, h);
            printf("layer=%d: LayerNorm 1 success!\n", l);

            // Multi-Head Attention
            //    a. QKV Projection
            //    b. Attention Mechanism (Softmax(Q*K^T / sqrt(d_k)) * V)
            //    c. Output Projection

            attention(h, num_tokens, transformer, attn_out);
            printf("layer=%d: Attention success!\n", l);

            // attention_cuda();

            // Residual Connection 1
            residual_connection(token_embeddings, attn_out, num_tokens);
            printf("layer=%d: Residual Connection success!\n", l);

            // Layer Norm 2
            layer_norm(token_embeddings, num_tokens, &transformer->ln_2, h);
            printf("layer=%d: LayerNorm 2 success!\n", l);

            // Feed Forward Network (GELU activation)
            //    a. Linear -> GELU
            //    b. Linear
            feed_forward(h, num_tokens, transformer, mlp_out);
            printf("layer=%d: Feed Forward success!\n", l);

            // Residual Connection 2
            residual_connection(token_embeddings, mlp_out, num_tokens);
            printf("layer=%d: Residual Connection success!\n\n", l);
        }

        // Final Layer Norm
        layer_norm(token_embeddings, num_tokens, &model->ln_f, h);
        printf("Final Layer norm success!\n");

        // Logits (MatMul with embedding table)
        logits(&h[(num_tokens - 1) * N_EMBD], d_logits, model->wte);
        printf("Logits success!\n");

        checkCudaErrors(cudaMemcpy(h_logits, d_logits, sizeof(float) * VOCAB_SIZE, cudaMemcpyDeviceToHost));

        top_k(h_logits, VOCAB_SIZE, 5, top_indices, top_scores);
        printf("Top K success!\n");

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
    cudaFree(d_logits);
    cudaFree(token_embeddings);
    cudaFree(d_tokens);
    return 0;
}