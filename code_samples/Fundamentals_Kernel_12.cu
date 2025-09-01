/*
Store the grid and block dimensions in `dim3` variables on the host before launching the kernel.
Launch the kernel using these variables, e.g., `myKernel<<<gridDim, blockDim>>>();`. This is the standard way to configure a launch.

The goal of this program is to demonstrate the standard CUDA launch configuration pattern:
1. Define `dim3` variables on the host for both the grid and block dimensions.
2. Use these variables in the kernel launch syntax `myKernel<<<gridDim, blockDim>>>();`.
3. Verify that the kernel executes correctly by performing a simple element-wise operation on a device array.

The program allocates an array on the device, initializes it on the host, copies it to the device,
launches a kernel that increments each element by one, copies the result back to the host,
and then verifies the output. Error checking is included to ensure each CUDA API call succeeds.
This pattern is widely used in CUDA programming for clarity and flexibility in launch configuration.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that adds 1 to each element of the array
__global__ void addOneKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Host memory allocation
    int *h_data = (int*)malloc(size);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    // Device memory allocation
    int *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(256);                 // 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x); // enough blocks to cover all elements

    // Launch kernel with the defined dimensions
    addOneKernel<<<gridDim, blockDim>>>(d_data, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != i + 1) {
            if (errors < 10) { // print first few errors
                fprintf(stderr, "Error at index %d: expected %d, got %d\n",
                        i, i + 1, h_data[i]);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("All values incremented correctly.\n");
    } else {
        printf("Found %d errors.\n", errors);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
