/*
Allocate an array of N structs on the device. Launch a kernel with N threads. Each thread i should modify the struct at d_data[i].

Thinking:
- Define a simple struct (e.g., containing an int id and a float value) to illustrate per-thread modifications.
- Allocate memory for N structs on the device using cudaMalloc.
- Launch a CUDA kernel with N threads (one block of N threads for simplicity, assuming N is small enough for a single block; otherwise we could compute grid dimensions).
- In the kernel, each thread computes its global index (i = blockIdx.x * blockDim.x + threadIdx.x) and checks bounds.
- Each thread then writes its id and some computed value (e.g., id*2.5f) into d_data[i].
- After kernel execution, copy the device array back to host memory to verify results.
- Add basic error checking for CUDA calls.
- Keep the code self-contained and compile-ready for nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Define a simple struct with two members.
struct MyStruct {
    int id;
    float value;
};

// CUDA kernel that modifies each struct at its index.
__global__ void modifyStructs(MyStruct* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // For demonstration, set id to idx and value to idx * 2.5
        d_data[idx].id = idx;
        d_data[idx].value = idx * 2.5f;
    }
}

// Helper macro for CUDA error checking.
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const int N = 10; // Number of structs

    // Allocate device memory for N structs
    MyStruct* d_data = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_data, N * sizeof(MyStruct)));

    // Launch kernel: one block with N threads (for simplicity)
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    modifyStructs<<<blocks, threadsPerBlock>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Allocate host memory to copy back results
    MyStruct h_data[N];
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(MyStruct), cudaMemcpyDeviceToHost));

    // Print results
    printf("Resulting structs on host:\n");
    for (int i = 0; i < N; ++i) {
        printf("Struct %d: id=%d, value=%f\n", i, h_data[i].id, h_data[i].value);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
