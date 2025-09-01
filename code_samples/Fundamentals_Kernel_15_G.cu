/*
Aim of the program: Launch a normal, fast kernel, followed immediately by the hanging kernel. What is the program's behavior?

Thinking:
1. The program should illustrate the effect of launching a kernel that never terminates (the hanging kernel) immediately after a normal, fast kernel.
2. The normal kernel will perform trivial work and finish quickly. We will use cudaDeviceSynchronize after launching it to ensure it completes before moving on.
3. Immediately after the fast kernel, we launch a second kernel that contains an infinite loop. This kernel will never return control to the host once it starts executing.
4. Because CUDA operations are serialized on the same stream by default, any subsequent kernel launch or memory copy that occurs after the hanging kernel will be queued but will never be executed until the hanging kernel finishes (which it never does). Thus the program will hang at the point of synchronization.
5. To observe this, the host will print messages before and after each kernel launch. After launching the hanging kernel, we will attempt another kernel launch and a cudaDeviceSynchronize. The message after the synchronize will never be printed, demonstrating the hang.
6. Error checking will be added so that if the program does return, we can see any errors, but in this scenario it should not return.

The code below implements this logic using CUDA C++ with device printf for visibility.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple fast kernel that does trivial work
__global__ void fastKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        printf("Fast kernel started\n");
    }
    // Dummy computation
    float dummy = 0.0f;
    for (int i = 0; i < 10; ++i) {
        dummy += i;
    }
}

// Hanging kernel that never terminates
__global__ void hangingKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        printf("Hanging kernel started\n");
        // Infinite loop
        while (true) {
            // Do nothing
        }
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

int main() {
    // Launch the normal fast kernel
    fastKernel<<<1, 256>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Fast kernel finished\n");

    // Launch the hanging kernel
    hangingKernel<<<1, 256>>>();
    CUDA_CHECK(cudaGetLastError());
    printf("Hanging kernel launched\n");

    // Attempt to launch another kernel after the hanging one
    fastKernel<<<1, 256>>>();
    CUDA_CHECK(cudaGetLastError());
    printf("Second kernel launch attempted\n");

    // This synchronization will block forever because the hanging kernel never terminates
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("This message will never be printed due to the hanging kernel.\n");

    return 0;
}
