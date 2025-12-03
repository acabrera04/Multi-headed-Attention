// This file will contain CUDA kernels
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <unistd.h>
#include <sys/time.h>

#define MAX_GRID_DIM 1 << 12
#define MAX_BLOCK_DIM 1 << 10
#define BLOCK_DIM 16
#define THREADS 1 << 8
#define MAX_TILE_WIDTH 16

__global__ void mat_mult_cuda(int *d_a, int *d_b, int *d_c, int m, int n, int p, int tile_width)
{
    __shared__ int a_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ int b_shared[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    /* Fill this func */
    int ph, k;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * tile_width + ty;
    int col = blockIdx.x * tile_width + tx;

    int sum = 0;
    // TODO Modify for when the dimensions don't matched (n placeholders for everything currently)
    for (ph = 0; ph < ceil(n / (float)tile_width); ph++)
    {
        int a_col = ph * tile_width + tx;
        int b_row = ph * tile_width + ty;
        if (row < n && a_col < n)
            a_shared[ty][tx] = d_a[row * n + a_col];
        else
            a_shared[ty][tx] = 0;

        if (col < n && b_row < n)
            b_shared[ty][tx] = d_b[b_row * n + col];
        else
            b_shared[ty][tx] = 0;

        __syncthreads();

        for (k = 0; k < tile_width; k++)
        {
            sum += a_shared[ty][k] * b_shared[k][tx];
        }
        __syncthreads();
    }
    if (row < n && col < n)
        d_c[row * n + col] = sum;
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

__global__ void mat_add_cuda(int *d_a, int *d_b, int *d_c, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        d_c[row * m + col] = d_a[row * m + col] + d_b[row * m + col];
    }
}

__global__ void softmax_cuda(int *A, int m, int n)
{
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