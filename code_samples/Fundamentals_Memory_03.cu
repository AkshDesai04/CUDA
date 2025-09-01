/*
Aim of the program:
Write a program that allocates two integer arrays, `d_a` and `d_b`, on the device. Initialize `d_a` by copying data from the host. Then, use `cudaMemcpy` with the `cudaMemcpyDeviceToDevice` kind to copy the contents of `d_a` directly to `d_b` without involving the host.

Thought process:
- I need to allocate memory on the GPU for two integer arrays. Use `cudaMalloc`.
- Create a host array and fill it with some known data (e.g., sequential numbers).
- Copy the host data to the device array `d_a` with `cudaMemcpy` of kind `cudaMemcpyHostToDevice`.
- Use `cudaMemcpy` again, this time with kind `cudaMemcpyDeviceToDevice`, to copy from `d_a` to `d_b`. This ensures the copy happens entirely on the GPU.
- To verify the operation, copy `d_b` back to the host and print the contents.
- Include basic error checking after each CUDA call using a helper macro.
- Wrap the main logic in a `main` function.
- Compile with `nvcc` and run to confirm that `d_b` matches the original host data.
*/

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error in " << __FILE__          \
                      << ":" << __LINE__ << " : "              \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main() {
    const int N = 10;
    const int size = N * sizeof(int);

    // Host array
    int h_a[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i + 1;  // Fill with 1..N
    }

    // Device pointers
    int *d_a = nullptr;
    int *d_b = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));

    // Copy from host to device (d_a)
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // Copy from device to device (d_a -> d_b)
    CHECK_CUDA(cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice));

    // Copy back from device to host for verification
    int h_b[N];
    CHECK_CUDA(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Host array h_a: ";
    for (int i = 0; i < N; ++i) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Device array d_b copied back to host h_b: ";
    for (int i = 0; i < N; ++i) {
        std::cout << h_b[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}
