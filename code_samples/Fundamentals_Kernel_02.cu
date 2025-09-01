/*
Aim of the program:
Modify the previous 'Hello World' kernel to print a message from the GPU itself.
The single thread in the kernel should use `printf` to print 'Hello from the GPU!'.
Note that the output from the GPU might appear at a different time than host-side prints.

Thinking:
- Use a simple CUDA kernel with one block and one thread.
- Inside the kernel, call device printf to output the message.
- After kernel launch, call cudaDeviceSynchronize() to ensure that all device output is flushed before the host continues.
- Include basic error checking for CUDA API calls to catch launch or synchronization failures.
- Use standard <stdio.h> and <cuda_runtime.h>.
- Provide a small host print to demonstrate ordering differences.
- Compile with a recent CUDA compiler that supports device printf (e.g., sm_35 or higher).
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that prints from the GPU
__global__ void helloGPUKernel()
{
    // Only one thread needs to print
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("Hello from the GPU!\n");
    }
}

int main(void)
{
    // Launch kernel with 1 block of 1 thread
    helloGPUKernel<<<1, 1>>>();
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to flush device printf output
    CHECK_CUDA(cudaDeviceSynchronize());

    // Host-side message to illustrate ordering
    printf("Hello from the host!\n");

    return 0;
}
