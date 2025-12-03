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

__global__ void matMul(int *A, int *B, int *C, int m, int n, int p)
{
}

__global__ void matScalar(int *A, int m, int n, int scalar)
{
}

__global__ void softMax(int *A, int m, int n)
{
}

int attention_cuda(int *h_Q, int *h_K, int *h_V, int *h_output, int q_rows, int q_cols, int k_rows, int k_cols, int v_rows, int v_cols)
{
    int i, iter;
    struct timeval timecheck;
    long dev_start, dev_end, dev_elapsed;

    gettimeofday(&timecheck, NULL);
    dev_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    int *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(reinterpret_cast<void **>(&d_Q), sizeof(int) * q_rows * q_cols);
    cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(int) * k_rows * k_cols);
    cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(int) * v_rows * v_cols);
    // TODO Confirm that this is correct size
    cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(int) * q_rows * v_cols);

    return 1;
}