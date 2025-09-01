/*
Create a `__device__` function that takes a pointer as an argument.

Thinking:
- The goal is to illustrate a __device__ function that accepts a pointer parameter.
- We define a device function that writes a value through a pointer.
- A simple kernel will call this function.
- The host will allocate memory, launch the kernel, copy back the result, and print it.
- Error checking is included for robustness.
*/

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Device function that takes a pointer and writes a value to the location it points to
__device__ void write_value(int *ptr, int val) {
    *ptr = val;
}

// Kernel that calls the device function
__global__ void kernel(int *d_arr) {
    // For simplicity, use only one thread
    write_value(d_arr, 42);
}

int main() {
    const size_t N = 1;
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *d_arr = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Launch kernel
    kernel<<<1, 1>>>(d_arr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Value after kernel execution: %d\n", h_arr[0]);

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
