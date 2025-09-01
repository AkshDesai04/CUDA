```cpp
/*
Implement scalar subtraction: `output[i] = input[i] - scalar_value`.

Thinking process:
1. The goal is to perform a simple element-wise subtraction of a constant scalar from each element of an input array, storing the result in an output array.
2. CUDA is a natural fit for this data-parallel operation: each thread will handle one array element.
3. Steps to accomplish this:
   - Define a kernel that computes `output[idx] = input[idx] - scalar_value` for each thread index `idx`.
   - In the host code, allocate and initialize an input array of a reasonable size (e.g., 1024 elements).
   - Allocate device memory for input and output arrays.
   - Copy the host input array to the device.
   - Determine grid and block dimensions and launch the kernel.
   - Copy the result back to the host and optionally print a few values for verification.
4. Include basic error checking for CUDA API calls to ensure proper execution.
5. Keep the program self-contained and compile-ready with `nvcc`, no external dependencies beyond the standard CUDA runtime.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                          \
    do {                                                                           \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                     \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                                   \
        }                                                                          \
    } while (0)

// Kernel: subtract scalar from each element
__global__ void scalar_subtract(const float* input, float* output, float scalar_value, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] - scalar_value;
    }
}

int main() {
    const int N = 1024;               // Number of elements
    const float scalar_value = 3.5f;  // Scalar to subtract

    // Host memory allocation
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    if (!h_input || !h_output) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with some values
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i; // e.g., 0.0, 1.0, 2.0, ...
    }

    // Device memory allocation
    float *d_input = NULL;
    float *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy input from host to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    scalar_subtract<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar_value, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results of scalar subtraction (input - %f):\n", scalar_value);
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("input[%d] = %f, output[%d] = %f\n", i, h_input[i], i, h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return EXIT_SUCCESS;
}
```