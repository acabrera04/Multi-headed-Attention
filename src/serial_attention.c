#ifndef M_PI
#define M_PI 3.1415926535897932384626433832
#endif

#include "load_tokens.h"
#include "model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void matMulSerial(float *A, float *B, float *C, int m, int n, int p);
void softMaxSerial(float *A, int m, int n);
void matScalar(float *A, int m, int n, float scalar);
void layerNormSerial(float *A, float *gamma, float *beta, float *output, int n);
void geluSerial(float *A, int m, int n);

// entry point
void attentionSerial(float *Q, float *K, float *V, float *output, int q_rows,
                     int q_cols, int k_rows, int k_cols, int v_rows,
                     int v_cols) {

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
void matMulSerial(float *A, float *B, float *C, int m, int n, int p) {
  int i, j, k;
  float sum;
  for (i = 0; i < m; i++) {
    for (j = 0; j < p; j++) {
      sum = 0.0f;
      for (k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * p + j];
      }
      C[i * p + j] = sum;
    }
  }
}

void matScalar(float *A, int m, int n, float scalar) {
  int i;
  for (i = 0; i < m * n; i++) {
    A[i] *= scalar;
  }
}

// this needs to be per row not per matrix
void softMaxSerial(float *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    float *row = &A[i * n];
    float max_val = row[0];
    for (int j = 1; j < n; j++) {
      if (row[j] > max_val)
        max_val = row[j];
    }

    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
      row[j] = exp(row[j] - max_val);
      sum += row[j];
    }

    for (int j = 0; j < n; j++) {
      row[j] /= sum;
    }
  }
}

void layerNormSerial(float *A, float *gamma, float *beta, float *output,
                     int n) {
  int i;
  float mean, variance;

  mean = 0.0f;
  variance = 0.0f;

  // calculate mean
  for (i = 0; i < n; i++) {
    mean += A[i];
  }
  mean /= n;

  // calculate variance
  for (i = 0; i < n; i++) {
    variance += (A[i] - mean) * (A[i] - mean);
  }
  variance /= n;

  // normalize
  for (i = 0; i < n; i++) {
    output[i] = gamma[i] * (A[i] - mean) / sqrt(variance + 1e-5f) + beta[i];
  }
}

// From https://www.baeldung.com/cs/gelu-activation-function
void geluSerial(float *A, int m, int n) {
  int i;
  for (i = 0; i < m * n; i++) {
    A[i] =
        0.5f * A[i] *
        (1.0f + tanh(sqrt(2.0f / M_PI) * (A[i] + 0.044715f * pow(A[i], 3.0f))));
  }
}

int main() {
  const char *model_path = "../work/gpt2_124m.bin";
  const char *tokens_path = "../work/tokens.bin";
  int num_tokens;

  GPT2Model *model = load_model_serial(model_path);
  int *tokens = load_tokens(tokens_path, &num_tokens);

  if (!(model && tokens)) {
    free_model_serial(model);
    free(tokens);
    return EXIT_FAILURE;
  }

  printf("model & input loaded\n");
}
