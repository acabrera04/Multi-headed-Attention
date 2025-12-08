// This file will contain helpers for CUDA

#define MAX_GRID_DIM (1 << 12)
#define MAX_BLOCK_DIM (1 << 10)
#define BLOCK_DIM 16
#define THREADS (1 << 8)
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