#ifndef M_PI
#define M_PI 3.1415926535897932384626433832
#endif

#include "load_tokens.h"
#include "model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void matMulSerial(float *A, float *B, float *C, int m, int n, int p);
void softMaxSerial(float *A, int m, int n);
void matScalar(float *A, int m, int n, float scalar);
void layerNormSerial(float *A, float *gamma, float *beta, float *output, int n);
void geluSerial(float *A, int m, int n);
void top_k(float *logits, int vocab_size, int k, int *top_indices, float *top_scores);

// C = A * B
// A: [M, K], B: [K, N], C: [M, N]
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

// this needs to be per row not per matrix
void softMaxSerial(float *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        float *row = &A[i * n];
        float max_val = row[0];
        for (int j = 1; j < n; j++)
        {
            if (row[j] > max_val)
                max_val = row[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < n; j++)
        {
            row[j] = exp(row[j] - max_val);
            sum += row[j];
        }

        for (int j = 0; j < n; j++)
        {
            row[j] /= sum;
        }
    }
}

void layerNormSerial(float *A, float *gamma, float *beta, float *output, int n)
{
    int i;
    float mean, variance;

    mean = 0.0f;
    variance = 0.0f;

    // calculate mean
    for (i = 0; i < n; i++)
    {
        mean += A[i];
    }
    mean /= n;

    // calculate variance
    for (i = 0; i < n; i++)
    {
        variance += (A[i] - mean) * (A[i] - mean);
    }
    variance /= n;

    // normalize
    for (i = 0; i < n; i++)
    {
        output[i] = gamma[i] * (A[i] - mean) / sqrt(variance + 1e-5f) + beta[i];
    }
}

// From https://www.baeldung.com/cs/gelu-activation-function
void geluSerial(float *A, int m, int n)
{
    int i;
    for (i = 0; i < m * n; i++)
    {
        A[i] = 0.5f * A[i] * (1.0f + tanh(sqrt(2.0f / M_PI) * (A[i] + 0.044715f * pow(A[i], 3.0f))));
    }
}

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

// now the actual attention function
void attentionSerial(float *x, TransformerBlock *block, int num_tokens, float *out)
{
    int head_dim = N_EMBD / N_HEAD; // split embedding size across heads (768 / 12 = 64)

    // allocate memory for qkv matrix
    float *qkv = (float *)malloc(num_tokens * 3 * N_EMBD * sizeof(float));

    // do the matrix multiplication and then add the bias and store in qkv matrix
    matMulSerial(x, block->c_attn.weight, qkv, num_tokens, N_EMBD, 3 * N_EMBD);

    for (int i = 0; i < num_tokens; i++)
    {
        for (int j = 0; j < 3 * N_EMBD; j++)
        {
            qkv[i * 3 * N_EMBD + j] += block->c_attn.bias[j];
        }
    }

    // decompose qkv matrix into Q, K, V
    float *q = (float *)malloc(N_HEAD * num_tokens * head_dim * sizeof(float));
    float *k = (float *)malloc(N_HEAD * num_tokens * head_dim * sizeof(float));
    float *v = (float *)malloc(N_HEAD * num_tokens * head_dim * sizeof(float));

    for (int token = 0; token < num_tokens; token++)
    {
        for (int head = 0; head < N_HEAD; head++)
        {
            for (int i = 0; i < head_dim; i++)
            {
                // index generated bc this is a 3d matrix flattened to 1d with dimensions: [N_HEAD, num_tokens, head_dim]
                int idx = num_tokens * head_dim * head + head_dim * token + i;

                int qkv_offset = token * 3 * N_EMBD;

                // Q is first N_EMBD, K is second N_EMBD, V is third N_EMBD (so offset by N_EMBD and 2*N_EMBD)
                q[idx] = qkv[qkv_offset + head * head_dim + i];
                k[idx] = qkv[qkv_offset + N_EMBD + head * head_dim + i];
                v[idx] = qkv[qkv_offset + 2 * N_EMBD + head * head_dim + i];
            }
        }
    }
    free(qkv);

    // now computing attention scores = dot product between q and k.T
    // q: [N_HEAD, num_tokens, head_dim]
    // k: [N_HEAD, num_tokens, head_dim]
    // leads to matrix of size: [N_HEAD, num_tokens, num_tokens], the attention matrix

    float *att = (float *)malloc(N_HEAD * num_tokens * num_tokens * sizeof(float));
    float scale = 1.0f / sqrt(head_dim); // and we scale it down

    // now compute dot products
    for (int head = 0; head < N_HEAD; head++)
    {
        for (int token = 0; token < num_tokens; token++)
        {
            for (int token2 = 0; token2 < num_tokens; token2++)
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
    }
    free(q);
    free(k);
    // v is still needed later

    // causal mask (a lower triangular matrix)
    // so that the model cannot attend to future tokens
    // basically this means setting upper triangular values to -inf (we use a large negative number)
    // whcih will be zeroed out in softmax
    for (int head = 0; head < N_HEAD; head++)
    {
        for (int token = 0; token < num_tokens; token++)
        {
            for (int token2 = 0; token2 < num_tokens; token2++)
            {
                if (token2 > token)
                {
                    att[head * num_tokens * num_tokens + token * num_tokens + token2] = -1e10f;
                }
            }
        }
    }

    // softmax
    // since its a 3D matrix flattened to 1D, we can just treat it as a 2D matrix of size:
    // [N_HEAD * num_tokens, num_tokens] to apply softmax row-wise
    softMaxSerial(att, N_HEAD * num_tokens, num_tokens);

    // now compute output values for each head with the attention weights and dot product with v
    // reminder: v is from qkv matrix
    // att: [N_HEAD, num_tokens, num_tokens]
    // v: [N_HEAD, num_tokens, head_dim]
    // y: [N_HEAD, num_tokens, head_dim]

    float *y_heads = (float *)malloc(N_HEAD * num_tokens * head_dim * sizeof(float));

    for (int head = 0; head < N_HEAD; head++)
    {
        for (int token = 0; token < num_tokens; token++)
        {
            for (int i = 0; i < head_dim; i++)
            {
                float sum = 0.0f;
                for (int j = 0; j < num_tokens; j++)
                {
                    sum += att[head * num_tokens * num_tokens + token * num_tokens + j] *
                           v[head * num_tokens * head_dim + j * head_dim + i];
                }
                y_heads[head * num_tokens * head_dim + token * head_dim + i] = sum;
            }
        }
    }
    free(att); // and now we are done with this attention matrix
    free(v);   // and v is no longer needed

    // now we reassemble the heads back into a single matrix
    // y_heads: [N_HEAD, num_tokens, head_dim] -> [num_tokens, N_EMBD]
    // the single matrix is what we got as input to the attention function, adjusted by this layer's attention
    float *y_reassembled = (float *)malloc(num_tokens * N_EMBD * sizeof(float));

    for (int i = 0; i < num_tokens; i++)
    {
        for (int h = 0; h < N_HEAD; h++)
        {
            for (int d = 0; d < head_dim; d++)
            {
                y_reassembled[i * N_EMBD + h * head_dim + d] = y_heads[h * num_tokens * head_dim + i * head_dim + d];
            }
        }
    }
    free(y_heads);

    // and now project back again with provided weights and bias from the model
    // y = y_reassembled dot c_proj_w + c_proj_b
    matMulSerial(y_reassembled, block->c_proj.weight, out, num_tokens, N_EMBD, N_EMBD);

    for (int i = 0; i < num_tokens; i++)
    {
        for (int j = 0; j < N_EMBD; j++)
        {
            out[i * N_EMBD + j] += block->c_proj.bias[j];
        }
    }
    free(y_reassembled);
}

// feedforward network (multilayer perceptron)
void mlpSerial(float *x, TransformerBlock *block, int num_tokens, float *out)
{
    // x: [num_tokens, N_EMBD]
    // c_fc: [N_EMBD, 4*N_EMBD]

    // hidden layer: [num_tokens, 4*N_EMBD]
    float *h = (float *)malloc(num_tokens * 4 * N_EMBD * sizeof(float));
    matMulSerial(x, block->c_fc.weight, h, num_tokens, N_EMBD, 4 * N_EMBD);

    // add bias
    for (int i = 0; i < num_tokens; i++)
    {
        for (int j = 0; j < 4 * N_EMBD; j++)
        {
            h[i * 4 * N_EMBD + j] += block->c_fc.bias[j];
        }
    }

    geluSerial(h, num_tokens, 4 * N_EMBD);

    // c_proj: [4*N_EMBD, N_EMBD]
    // now project based on model weights
    matMulSerial(h, block->c_proj_mlp.weight, out, num_tokens, 4 * N_EMBD, N_EMBD);

    // add bias
    for (int i = 0; i < num_tokens; i++)
    {
        for (int j = 0; j < N_EMBD; j++)
        {
            out[i * N_EMBD + j] += block->c_proj_mlp.bias[j];
        }
    }
    free(h);
}

// process a single transformer block
void transformerBlockSerial(float *x, TransformerBlock *block, int num_tokens)
{
    float *h = (float *)malloc(num_tokens * N_EMBD * sizeof(float));
    float *attn_out = (float *)malloc(num_tokens * N_EMBD * sizeof(float));

    // layer norm 1 - normalize each token embedding
    for (int i = 0; i < num_tokens; i++)
    {
        layerNormSerial(&x[i * N_EMBD], block->ln_1.weight, block->ln_1.bias, &h[i * N_EMBD], N_EMBD);
    }

    attentionSerial(h, block, num_tokens, attn_out);

    // residual
    for (int i = 0; i < num_tokens * N_EMBD; i++)
    {
        x[i] += attn_out[i];
    }

    // layer norm 2 - normalize each token embedding
    for (int i = 0; i < num_tokens; i++)
    {
        layerNormSerial(&x[i * N_EMBD], block->ln_2.weight, block->ln_2.bias, &h[i * N_EMBD], N_EMBD);
    }

    // feed forward network
    float *mlp_out = (float *)malloc(num_tokens * N_EMBD * sizeof(float));
    mlpSerial(h, block, num_tokens, mlp_out);

    // residual
    for (int i = 0; i < num_tokens * N_EMBD; i++)
    {
        x[i] += mlp_out[i];
    }

    free(h);
    free(attn_out);
    free(mlp_out);
}

int main()
{
    const char *model_path = "./work/gpt2_124m.bin";
    const char *tokens_path = "./work/tokens.bin";
    const char *output_path = "./work/serial_output.bin";

    GPT2Model *model = load_model_serial(model_path);
    if (!model)
        return 1;

    int num_tokens;
    int *tokens = load_tokens(tokens_path, &num_tokens);
    if (!tokens)
        return 1;

    struct timeval timecheck;

    long dev_start, dev_end, dev_elapsed;
    gettimeofday(&timecheck, NULL);
    dev_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    // input embedding
    float *x = (float *)malloc(num_tokens * N_EMBD * sizeof(float));

    for (int i = 0; i < num_tokens; i++)
    {
        int token = tokens[i];
        for (int j = 0; j < N_EMBD; j++)
        {
            // add token embedding and position embedding (wte + wpe)
            x[i * N_EMBD + j] = model->wte[token * N_EMBD + j] + model->wpe[i * N_EMBD + j];
        }
    }

    // iterate over transformer blocks
    for (int i = 0; i < N_LAYER; i++)
    {
        transformerBlockSerial(x, &model->h[i], num_tokens);
    }

    // final layer norm
    float *x_final = (float *)malloc(num_tokens * N_EMBD * sizeof(float));
    for (int i = 0; i < num_tokens; i++)
    {
        layerNormSerial(&x[i * N_EMBD], model->ln_f.weight, model->ln_f.bias, &x_final[i * N_EMBD], N_EMBD);
    }
    free(x);

    // generate next token logits (logit are scores for each token in vocab)
    // logits = x_final dot wte.T

    // we look at the last token's embedding to generate the next token
    float *last_token_emb = &x_final[(num_tokens - 1) * N_EMBD];
    float *logits = (float *)malloc(VOCAB_SIZE * sizeof(float));

    // compute the logits for all tokens in the vocab (their scores)
    for (int i = 0; i < VOCAB_SIZE; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < N_EMBD; j++)
        {
            sum += last_token_emb[j] * model->wte[i * N_EMBD + j];
        }
        logits[i] = sum;
    }

    // top 5 selection
    int top_indices[5];
    float top_scores[5];
    top_k(logits, VOCAB_SIZE, 5, top_indices, top_scores);

    gettimeofday(&timecheck, NULL);
    dev_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
    dev_elapsed = dev_end - dev_start;

    printf("rank=%d: DEV time: %d procs: %ld msecs\n", 0, 1, dev_elapsed);

    printf("\nTop 5 predictions for the next token:\n");
    for (int i = 0; i < 5; i++)
    {
        printf("%d. Token ID: %d, Score: %.4f\n", i + 1, top_indices[i], top_scores[i]);
    }

    // save output predictions to file
    FILE *f = fopen(output_path, "wb");
    fwrite(top_indices, sizeof(int), 5, f);
    fclose(f);

    printf("\nTop 5 token IDs saved to %s\n", output_path);

    free(x_final);
    free(logits);
    free(tokens);
    free_model_serial(model);

    return 0;
}
