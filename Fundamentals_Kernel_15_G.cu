/*
Launch a normal, fast kernel, followed immediately by the hanging kernel. What is the program's behavior?

Thinking:
- The program demonstrates the effect of launching a normal quick kernel followed by a kernel that never terminates (hangs).
- In CUDA, kernel launches are asynchronous with respect to the host but are serialized on the default stream (stream 0).
- The first kernel (fastKernel) runs quickly and finishes.
- The second kernel (hangingKernel) contains an infinite loop and never returns.
- Since both kernels are launched to the same default stream, the second kernel starts executing only after the first finishes.
- After launching both kernels, the host attempts to copy data back from device memory using cudaMemcpy, which implicitly synchronizes the default stream.
- The cudaMemcpy call will block indefinitely because the hanging kernel is still executing and has not completed.
- Thus, the program stalls at the cudaMemcpy, never reaching the point where it prints the result or exits normally.
- This behavior illustrates that hanging kernels can cause the host program to hang if they are launched on the same stream and a subsequent sync or copy is performed.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// A normal fast kernel that writes a value to global memory
__global__ void fastKernel(int *d_arr) {
    int idx = threadIdx.x;
    if (idx < 1) {
        d_arr[idx] = 42; // write a known value
    }
}

// A hanging kernel that never terminates (infinite loop)
__global__ void hangingKernel(int *d_arr) {
    int idx = threadIdx.x;
    // This loop will never exit; it effectively hangs the GPU
    while (true) {
        // Do nothing, just spin
        if (idx < 1) {
            d_arr[idx] = -1; // just to prevent compiler optimizing away
        }
    }
}

int main() {
    const int N = 1;
    int *d_arr = nullptr;
    int h_arr[N];

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    // Launch the fast kernel
    fastKernel<<<1, N>>>(d_arr);
    CUDA_CHECK(cudaGetLastError());

    // Launch the hanging kernel immediately after
    hangingKernel<<<1, N>>>(d_arr);
    CUDA_CHECK(cudaGetLastError());

    // Attempt to copy data back from device to host
    // This call will block forever because the hanging kernel has not finished
    printf("Attempting to copy data from device to host...\n");
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Copy completed. Value in array: %d\n", h_arr[0]);

    // Clean up (this point will never be reached due to the hanging kernel)
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
