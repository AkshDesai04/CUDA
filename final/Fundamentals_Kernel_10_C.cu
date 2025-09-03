/*
Aim of the program:
Pass an array of these structs from host to device. Have each thread `i` access the `i`-th element of the struct array and print a member.

Thinking:
- We need a concrete struct definition to work with. I'll define a simple struct named `MyStruct` containing an integer `id` and a float `value`. This struct will be used to create an array on the host.
- The host will allocate an array of `MyStruct`, initialize it with sample data, then allocate a corresponding array on the device with `cudaMalloc`.
- Use `cudaMemcpy` to copy the host array to device memory.
- Launch a CUDA kernel with enough threads to cover each element. Each thread will compute its global index `idx`. If `idx` is within bounds, the thread will use `printf` (device-side printing) to output its `id` and `value`.
- Since device printf requires certain compile flags, but NVCC handles it automatically when compiled with `-arch=sm_xx`, we just need to include `#include <stdio.h>` and compile with a recent toolkit.
- We'll handle basic error checking for CUDA API calls.
- After kernel execution, we will synchronize, free device memory, and exit.

The program is self-contained and can be compiled with:
    nvcc -o struct_print struct_print.cu
and run with:
    ./struct_print
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Define the struct to be passed from host to device
struct MyStruct {
    int id;
    float value;
};

// Kernel that prints a member of each struct element
__global__ void printStructs(const MyStruct* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Device-side printf requires that the device supports printf
        printf("Thread %d: id=%d, value=%f\n", idx, arr[idx].id, arr[idx].value);
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main() {
    const int N = 10;  // Number of struct elements

    // Allocate and initialize host array
    MyStruct* h_arr = (MyStruct*)malloc(N * sizeof(MyStruct));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_arr[i].id = i;
        h_arr[i].value = i * 1.1f;
    }

    // Allocate device array
    MyStruct* d_arr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(MyStruct)));

    // Copy host array to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(MyStruct), cudaMemcpyHostToDevice));

    // Launch kernel: one thread per struct element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printStructs<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
