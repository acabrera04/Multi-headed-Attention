/*
  Allen Cabrera, Avanish Kulkarni
  This file will contain the driver to run the CUDA kernels on the device
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_STEPS 100
#define N_LAYER 12

// Define model struct

// Define run state (buffers for activations)

// Load weights from file

// Initialize CUDA device

// Encode input data

int main(void) {
  // Initialize - start CUDA device, load model, allocate memory, create current state

  // Input Processing - encode/tokenize input text, get tokens

  for (int step = 0; step < NUM_STEPS; ++step) {
    // Launch CUDA kernel for inference
    
    // Lookup embedding
    
    // Transformer Layers
    for (int l = 0; l < N_LAYER; l++) {
      // Layer Norm 1
      
      // Multi-Head Attention
      //    a. QKV Projection
      //    b. Attention Mechanism (Softmax(Q*K^T / sqrt(d_k)) * V)
      //    c. Output Projection
      
      // Residual Connection 1
      
      // Layer Norm 2
      
      // Feed Forward Network (GELU activation)
      //    a. Linear -> GELU
      //    b. Linear
      
      // Residual Connection 2
    }

    // Final Layer Norm

    // Logits (MatMul with embedding table) 
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    // Next Token Selection
    // printf("%s", decode_token(current_token));
  }

  // Free memory
    
  return 0;
}