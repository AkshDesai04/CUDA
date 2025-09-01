```cpp
/* 
Aim of the program: Write a program that allocates an array of 256 integers on the host and initializes it with values from 0 to 255. Then, allocate memory for 256 integers on the GPU device using `cudaMalloc`. Finally, copy the host array to the device array using `cudaMemcpy` with the `cudaMemcpyHostToDevice` kind.

Thinking:
- Use the CUDA Runtime API (`cuda_runtime.h`) for allocation, copying, and error checking.
- Allocate a host array of size 256 (`int* h_arr = new int[256];`) and initialize it with a simple loop.
- Allocate device memory for 256 integers with `cudaMalloc(&d_arr, 256 * sizeof(int));`.
- Copy the data from host to device with `cudaMemcpy(d_arr, h_arr, 256 * sizeof(int), cudaMemcpyHostToDevice);`.
- Perform error checks after each CUDA call to catch any problems.
- After the copy, free both host and device memory with `delete[]` and `cudaFree`.
- Provide minimal console output to indicate success.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void) {
    const int N = 256;
    int* h_arr = new int[N];
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with values 0 to 255
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    int* d_arr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    printf("Successfully copied %d integers from host to device.\n", N);

    // Clean up
    delete[] h_arr;
    CUDA_CHECK(cudaFree(d_arr));

    return EXIT_SUCCESS;
}
```