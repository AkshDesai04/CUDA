```cuda
/*
Write a kernel that does an infinite loop (`while(1);`). Launch it. Observe that your program hangs. This demonstrates the asynchronous nature of kernel launches and the importance of `cudaDeviceSynchronize` for debugging.

Thinking:
- The goal is to create a CUDA program that launches a kernel which never terminates.
- The kernel will contain an infinite loop: `while (1);` (or `while(true);`).
- The main host code will launch this kernel without any subsequent synchronization.
- Because CUDA kernel launches are asynchronous, the host will return immediately from the launch and proceed to the end of main.
- Since the kernel never terminates, the GPU will be busy forever, but the host thread may exit or hang depending on the environment.
- To illustrate the asynchronous nature and the role of `cudaDeviceSynchronize`, we will comment out the call to `cudaDeviceSynchronize`. If we uncomment it, the host will wait indefinitely, demonstrating that the kernel is still running.
- The program will also include basic error checking to illustrate best practices, though it is not strictly necessary for the demonstration.
- When run, the program should hang (or print a message and then hang if we include a print after the launch), showing that the kernel never completes.
- The code uses `__global__` for the kernel and simple launch parameters (e.g., 1 block, 1 thread) because only the infinite loop matters.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel that runs an infinite loop
__global__ void infinite_loop_kernel() {
    // Each thread will spin forever
    while (1) {
        // Do nothing; just loop
    }
}

int main(void) {
    printf("Launching infinite loop kernel.\n");

    // Launch the kernel with 1 block of 1 thread
    infinite_loop_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());

    // Uncomment the next line to wait for the kernel to finish (it will never finish).
    // CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel launch complete (host thread continues).\n");

    // Since the kernel never terminates, the program will hang here if we wait.
    // We return 0 to indicate normal exit if the program were to finish.
    return 0;
}
```