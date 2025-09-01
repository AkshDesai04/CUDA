```cpp
/*
Aim of the program: Pass an array of these structs from host to device. Have each thread `i` access the `i`-th element of the struct array and print a member.

Thinking:
1. Define a simple struct (e.g., MyStruct) with a few members that are easily printable, such as an integer ID and a floating-point value.
2. Create an array of this struct on the host, initialize it with sample data.
3. Allocate memory for the same array on the device.
4. Copy the host array to the device using cudaMemcpy.
5. Write a kernel where each thread calculates its global index `i` and accesses `arr[i]`. If `i` is within bounds, the thread prints one of the struct's members using `printf`. Printing from device is supported with `printf` when the GPU device has the capability and the kernel launch includes a sufficient number of threads.
6. Launch the kernel with enough blocks and threads so that each struct element is processed by a unique thread.
7. After kernel execution, synchronize, free device memory, and exit.
8. Add basic error checking after each CUDA API call to make debugging easier.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct MyStruct {
    int id;
    float value;
};

__global__ void printStructs(const MyStruct* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Print the id and value of the idx-th struct
        printf("Thread %d: id=%d, value=%f\n", idx, arr[idx].id, arr[idx].value);
    }
}

int main() {
    const int N = 10;                    // Number of struct elements
    const int THREADS_PER_BLOCK = 256;   // Adjust as needed

    // Allocate and initialize host array
    MyStruct* h_array = (MyStruct*)malloc(N * sizeof(MyStruct));
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_array[i].id = i;
        h_array[i].value = (float)i * 1.1f;
    }

    // Allocate device memory
    MyStruct* d_array = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_array, N * sizeof(MyStruct));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_array);
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_array, h_array, N * sizeof(MyStruct), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        free(h_array);
        return EXIT_FAILURE;
    }

    // Determine grid and block dimensions
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    printStructs<<<blocks, THREADS_PER_BLOCK>>>(d_array, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        free(h_array);
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        free(h_array);
        return EXIT_FAILURE;
    }

    // Clean up
    cudaFree(d_array);
    free(h_array);

    printf("All done.\n");
    return EXIT_SUCCESS;
}
```