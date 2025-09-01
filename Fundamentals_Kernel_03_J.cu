/*
Aim: Launch with one block of 32 threads (one warp). Have thread 0 print a message. Then have thread 15 print a message. Then thread 31.

Thinking:
- We need a CUDA kernel that runs on a single block of 32 threads.
- Each thread will check its threadIdx.x value and print a message only if it matches one of the required indices (0, 15, 31).
- To enforce the order of printing (first thread 0, then thread 15, then thread 31) we use __syncthreads() barriers between the print statements. This ensures that all threads reach the barrier before any can proceed to the next print, guaranteeing that the outputs appear in the requested sequence.
- Device printf is used for printing from device code. It works on compute capability >= 2.0, which is the typical case.
- The host code launches the kernel with <<<1,32>>> and then synchronizes the device to wait for kernel completion before exiting.
- A simple error-checking macro is included for CUDA API calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                     \
            return EXIT_FAILURE;                                  \
        }                                                         \
    } while (0)

__global__ void printKernel()
{
    // Thread 0 prints first
    if (threadIdx.x == 0) {
        printf("Message from thread 0\n");
    }
    __syncthreads(); // Ensure thread 0 finishes before others proceed

    // Thread 15 prints next
    if (threadIdx.x == 15) {
        printf("Message from thread 15\n");
    }
    __syncthreads(); // Ensure thread 15 finishes before others proceed

    // Thread 31 prints last
    if (threadIdx.x == 31) {
        printf("Message from thread 31\n");
    }
}

int main(void)
{
    // Launch kernel with 1 block of 32 threads
    printKernel<<<1, 32>>>();
    CHECK_CUDA(cudaGetLastError());

    // Wait for kernel to finish and capture any errors
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset device (optional, but good practice)
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
