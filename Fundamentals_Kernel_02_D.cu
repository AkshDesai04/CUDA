```cuda
/* 
Aim of the program which is passed in this prompt word to word: 
"Modify the printf to include the block's ID: printf("Hello from block %d!\n", blockIdx.x);. Launch with 8 blocks."

Thinking: 
The task requires a simple CUDA kernel that prints a message from each block, including the block's ID. 
We modify the printf statement to embed blockIdx.x. We launch the kernel with 8 blocks, each having one thread, 
since the message depends only on the block index. 
We use the device runtime printf capability, so we need to compile with nvcc and enable printf on device. 
We also add a basic error-checking macro for CUDA API calls, and wait for device completion before exiting. 
The kernel does not need any shared or global memory, just a printf. 
The host code will launch the kernel with <<<8, 1>>> and then call cudaDeviceSynchronize().
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Kernel that prints a message including the block's ID
__global__ void helloKernel() {
    // Each block prints its own ID
    printf("Hello from block %d!\n", blockIdx.x);
}

int main() {
    // Launch the kernel with 8 blocks and 1 thread per block
    helloKernel<<<8, 1>>>();
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Wait for the kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
```