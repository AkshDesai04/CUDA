```cuda
/*
Aim of the program: 
Modify the kernel so that only threads with an even index print their ID. 

Thinking process:
1. The requirement is to alter a CUDA kernel so that only threads whose global index
   (e.g., threadIdx.x + blockIdx.x * blockDim.x) is even will output their ID.
2. Device-side printing in CUDA is done with printf. 
3. To keep the example minimal, the program will:
   - Define a kernel that calculates a global thread index.
   - Check if the index is even; if so, call printf to print the ID.
4. Host code will launch the kernel with a configurable number of blocks and threads,
   then synchronize and return.
5. Error checking macros will be added for robustness.
6. Since printf is used, we need to compile for a compute capability that supports it
   (≥ sm_20) which is the default for most GPUs.
7. The final code will be a single .cu file with the comment block at the top
   containing the aim and this reasoning, followed by the full program.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel: each thread computes its global index and prints it if even
__global__ void printEvenThreadIds()
{
    // Compute 1D global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only even indices should print
    if ((idx & 1) == 0) { // bitwise AND is faster than modulo
        printf("Thread ID: %d\n", idx);
    }
}

int main()
{
    // Parameters: number of threads per block and number of blocks
    const int threadsPerBlock = 8; // can be tuned
    const int numBlocks = 4;       // can be tuned

    // Launch kernel
    printEvenThreadIds<<<numBlocks, threadsPerBlock>>>();
    CUDA_CHECK(cudaGetLastError());

    // Ensure all printf output is flushed
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up and exit
    return 0;
}
```