#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void matMulSerial(float *A, float *B, float *C, int m, int n, int p);
void softMaxSerial(float *A, int m, int n);
void matScalar(float *A, int m, int n, float scalar);

// entry point
void attentionSerial(float *Q, float *K, float *V, float *output, int q_rows, int q_cols, int k_rows, int k_cols, int v_rows, int v_cols)
{
    // figure out how big this should be
    int buffer_rows = q_rows, buffer_cols = k_rows;
    float *buffer = malloc(sizeof(float) * buffer_rows * buffer_cols);

    // Q*K^T
    matMulSerial(Q, K, buffer, q_rows, q_cols, k_rows);

    // get sqrt of d_k and divide buffer by that
    // d_k = # of columns in queries/keys
    matScalar(buffer, buffer_rows, buffer_cols, 1 / sqrt(q_cols));

    // softmax
    softMaxSerial(buffer, buffer_rows, buffer_cols);

    // softmax * V
    matMulSerial(buffer, V, output, buffer_rows, buffer_cols, v_cols);

    free(buffer);
}

// TODO Add transpose
void matMulSerial(float *A, float *B, float *C, int m, int n, int p)
{
    int i, j, k;
    float sum;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < p; j++)
        {
            sum = 0.0f;
            for (k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void matScalar(float *A, int m, int n, float scalar)
{
    int i;
    for (i = 0; i < m * n; i++)
    {
        A[i] *= scalar;
    }
}

void softMaxSerial(float *A, int m, int n)
{
    float sum_exp = 0.0;
    int i = 0;
    for (i = 0; i < m * n; i++)
    {
        A[i] = exp(A[i]);

        sum_exp += A[i];
    }

    for (i = 0; i < m * n; i++)
    {
        A[i] /= sum_exp;
    }
}

void layerNormSerial(float *input, float *gamma, float *beta, float *output, int n)
{
    int i;
    float mean, variance;

    mean = 0.0f;
    variance = 0.0f;

    // calculate mean
    for (i = 0; i < n; i++)
    {
        mean += input[i];
    }
    mean /= n;

    // calculate variance
    for (i = 0; i < n; i++)
    {
        variance += (input[i] - mean) * (input[i] - mean);
    }
    variance /= n;

    // normalize
    for (i = 0; i < n; i++)
    {
        output[i] = gamma[i] * (input[i] - mean) / sqrt(variance + 1e-5f) + beta[i];
    }
}