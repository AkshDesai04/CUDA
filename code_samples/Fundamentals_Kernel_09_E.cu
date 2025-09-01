/*
Create a `__host__ __device__` function (a function that can be called from both host and device). Call it from both `main` and your kernel.

Thinking:
- Define a simple arithmetic function `add` that can run on both host and device. 
- Use the CUDA qualifiers `__host__ __device__` so the compiler generates two versions of the function. 
- In `main`, call `add` directly and print the result to demonstrate host usage. 
- Create a kernel that also calls `add` for each thread and writes the result to a device array. 
- After kernel launch, copy the results back to host memory and print them. 
- Add minimal CUDA error checking for robustness. 
- Keep the code self‑contained so it can be compiled with `nvcc` and run on a GPU.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define N 10

// Function that can run on both host and device
__host__ __device__
int add(int a, int b) {
    return a + b;
}

// Kernel that uses the host/device function
__global__
void kernel(int *d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = add(idx, idx + 1); // Call the dual‑qualified function
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main() {
    // Host call to the function
    int host_val = add(3, 4);
    printf("Host call to add(3,4) = %d\n", host_val);

    // Allocate device memory
    int *d_out;
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(int)));

    // Launch kernel
    int threadsPerBlock = 4;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocks, threadsPerBlock>>>(d_out);
    CHECK_CUDA(cudaGetLastError());

    // Copy results back to host
    int h_out[N];
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print kernel results
    printf("Kernel results:\n");
    for (int i = 0; i < N; ++i) {
        printf("add(%d,%d) = %d\n", i, i + 1, h_out[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
