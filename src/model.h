#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

// hardcode gpt2 parameters
#define N_LAYER 12
#define N_EMBD 768
#define N_HEAD 12
#define VOCAB_SIZE 50257
#define MAX_POS_EMBD 1024
#define CTX_LEN 1024

// a layer normalization operation stores weight and bias
// y = (x - mean(x)) / sqrt(var(x) + epsilon) * gamma + beta (where x is the input, gamma is the weight, beta is the bias)
typedef struct {
    float *weight;
    float *bias;
} LayerNorm;

// a linear operation stores weight and bias
// y = Wx + b (where x is the input, W is the weight, b is the bias)
typedef struct {
    float *weight;
    float *bias;
} Linear;

// a transformer block stores layer norm, linear operations, and positional embeddings
/* this mirrors gpt2's transformer block 
  1. layer norm
  2. linear layer that produces Q, K, V
  3. projection for attention
  4. layer norm again before mlp
  5. linear layer in feedforward for mlp
  6. second linear layer in mlp
*/
typedef struct {
    LayerNorm ln_1;
    Linear c_attn;
    Linear c_proj;
    LayerNorm ln_2;
    Linear c_fc;
    Linear c_proj_mlp;
} TransformerBlock;

// a model stores the token embeddings, positional embeddings, transformer blocks, and final layer norm
/* 
  1. token embeddings
  2. positional embeddings
  3. transformer blocks
  4. final layer norm
*/
typedef struct {
    float *wte;
    float *wpe;
    TransformerBlock h[N_LAYER];
    LayerNorm ln_f;
} GPT2Model;

GPT2Model* load_model(const char* filename);
void free_model(GPT2Model* model);

#endif // MODEL_H
