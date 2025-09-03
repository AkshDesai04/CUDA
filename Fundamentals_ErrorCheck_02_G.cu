/*
Aim of the program: Does `cudaDeviceSynchronize` need to be wrapped? Yes, it can return asynchronous errors.

Thinking:
1. The user is asking whether we need to check the return value of cudaDeviceSynchronize. In CUDA, many API calls return an error code, and cudaDeviceSynchronize can indeed report errors that were generated asynchronously during kernel execution (e.g., illegal memory access, launch failure). Therefore, it is good practice to check the return value of cudaDeviceSynchronize to catch such errors.
2. To illustrate this, we will write a minimal CUDA C program that:
   - Launches a simple kernel that intentionally performs an illegal memory access if a certain condition is met.
   - Calls cudaDeviceSynchronize() after the kernel launch.
   - Wraps cudaDeviceSynchronize in error checking, printing the error message if one occurs.
3. The program will include a helper macro `CHECK_CUDA(call)` that checks the return value of a CUDA API call and prints a helpful message.
4. The kernel will use a parameter to decide whether to cause an error. We will run the kernel twice: once normally, and once with the error condition enabled, to demonstrate that cudaDeviceSynchronize reports the error.
5. The code is self-contained and ready to compile with `nvcc`.

The result is a .cu file that demonstrates the necessity of wrapping cudaDeviceSynchronize for proper error handling.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Simple kernel that can optionally cause an illegal memory access
__global__ void testKernel(int *d_arr, int size, int cause_error) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (cause_error && idx == size - 1) {
            // Intentional out-of-bounds write to trigger an error
            d_arr[idx + 1] = 42;
        } else {
            d_arr[idx] = idx;
        }
    }
}

int main(void) {
    const int N = 256;
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *d_arr = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Launch kernel without error
    testKernel<<<(N + 31) / 32, 32>>>(d_arr, N, 0);
    // Wrap cudaDeviceSynchronize with error checking
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Kernel 1 executed successfully.\n");

    // Launch kernel with intentional error
    testKernel<<<(N + 31) / 32, 32>>>(d_arr, N, 1);
    // Wrap cudaDeviceSynchronize with error checking
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Kernel 2 executed successfully.\n");

    // Copy back data (this part won't be reached if the error occurs)
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
