```cuda
/*
Aim of the program:
Allocate an array of N structs on the device. Launch a kernel with N threads. Each thread i should modify the struct at d_data[i].

Thinking:
1. Define a simple struct (e.g., containing an int and a float) so we can easily observe modifications.
2. Allocate a host array of N structs and initialize it for clarity.
3. Allocate device memory for an array of N structs using cudaMalloc.
4. Copy the host array to device memory (optional for demonstration; we could also initialize directly on device).
5. Create a CUDA kernel that takes a pointer to the struct array and the number of elements.
   Inside the kernel, compute the global thread index i. If i < N, modify the struct at d_data[i].
   For demonstration, set the integer field to the thread index and the float field to a function of i.
6. Launch the kernel with one thread per struct: use a grid size of ceil(N / blockDim.x).
7. Copy the modified data back to the host and print a few elements to verify the changes.
8. Free device and host memory, handle errors appropriately.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Number of structs

// Simple struct definition
struct MyStruct {
    int   a;
    float b;
};

// CUDA error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that modifies each struct in the array
__global__ void modifyStructs(MyStruct *d_data, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Example modification: set 'a' to thread index and 'b' to a float value
        d_data[idx].a = idx;
        d_data[idx].b = 0.5f * (float)idx;
    }
}

int main(void) {
    // Host array allocation and initialization
    MyStruct *h_data = (MyStruct*)malloc(N * sizeof(MyStruct));
    for (int i = 0; i < N; ++i) {
        h_data[i].a = -1;           // placeholder
        h_data[i].b = -1.0f;        // placeholder
    }

    // Device array allocation
    MyStruct *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(MyStruct)));

    // Copy host data to device (not strictly necessary for this demo)
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(MyStruct), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    modifyStructs<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy modified data back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(MyStruct), cudaMemcpyDeviceToHost));

    // Print first 10 modified structs to verify
    printf("First 10 modified structs:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("Index %d: a = %d, b = %f\n", i, h_data[i].a, h_data[i].b);
    }

    // Clean up
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));

    // Reset device
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```