```cpp
/*
Aim of the program:
Allocate an array of N structs on the device. Launch a kernel with N threads. Each thread i should modify the struct at d_data[i].

Thinking process:
1. Define a simple struct `MyStruct` that contains two members (int id and float value). This struct will be stored in device memory.
2. Allocate memory on the device for an array of `N` such structs using `cudaMalloc`.
3. Write a CUDA kernel `modify_structs` that takes a pointer to the array. Each thread calculates its global index `i` (here we use a single block so the index is `threadIdx.x`). The thread writes its own id and a computed value into `d_data[i]`. This demonstrates that each thread modifies its own struct.
4. In the host code:
   - Set the number of elements `N`.
   - Allocate device memory.
   - Launch the kernel with `N` threads (one block). If N is large, we would compute grid/block dimensions accordingly, but for simplicity we assume it fits in one block.
   - Copy the array back to host memory using `cudaMemcpy`.
   - Print a few elements to verify the modifications.
5. Add basic CUDA error checking to make debugging easier.
6. Ensure the program compiles as a .cu file with `nvcc`.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple struct to be stored on the device
struct MyStruct {
    int id;
    float value;
};

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that modifies each struct in the array
__global__ void modify_structs(MyStruct* d_data, int N) {
    int i = threadIdx.x;          // Since we launch one block of N threads
    if (i < N) {
        d_data[i].id = i;                 // Set id to thread index
        d_data[i].value = i * 1.0f;        // Set value to thread index as float
    }
}

int main() {
    const int N = 10; // Number of structs
    MyStruct* d_data = nullptr;
    MyStruct* h_data = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(MyStruct)));

    // Allocate host memory for verification
    h_data = (MyStruct*)malloc(N * sizeof(MyStruct));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Launch kernel with N threads in a single block
    dim3 blockSize(N);
    dim3 gridSize(1);
    modify_structs<<<gridSize, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(MyStruct), cudaMemcpyDeviceToHost));

    // Print results
    for (int i = 0; i < N; ++i) {
        printf("Struct %d: id = %d, value = %f\n", i, h_data[i].id, h_data[i].value);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return 0;
}
```