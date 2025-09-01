```cuda
/* 
Implement for `double` data type. 
This program demonstrates a simple vector addition on the GPU using CUDA. 
The aim of the program is to take two input arrays of doubles on the host, copy them to the device, launch a kernel that adds corresponding elements, copy the result back, and finally verify the result on the host. 
The design follows the typical CUDA programming model: 
  1. Allocate host memory for input vectors A and B and output vector C. 
  2. Allocate device memory for d_A, d_B, and d_C. 
  3. Initialize input vectors with some sample data. 
  4. Copy input data from host to device. 
  5. Define a kernel that performs element-wise addition for each index. 
  6. Determine grid and block dimensions based on the size of the vectors. 
  7. Launch the kernel. 
  8. Copy the result back from device to host. 
  9. Verify the results by comparing each element of C with the expected sum. 
  10. Clean up memory. 
The program includes basic error checking for CUDA API calls and kernel launches to aid debugging. 
The kernel is written for double precision (64-bit) floating point values, which requires that the GPU support compute capability 1.3 or higher (most modern GPUs support this). 
The sample size is set to 1,024,000 elements to illustrate performance with a reasonably large vector, but the size can be easily modified. 
All code is self‑contained in a single .cu file and can be compiled with nvcc. 
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for checking CUDA errors following a CUDA API call or kernel launch
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel for element-wise addition of two double vectors
__global__ void vecAdd(const double* A, const double* B, double* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    // Size of vectors
    const int N = 1024 * 1024;  // 1,048,576 elements
    size_t size = N * sizeof(double);

    // Allocate host memory
    double *h_A = (double*)malloc(size);
    double *h_B = (double*)malloc(size);
    double *h_C = (double*)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<double>(i);
        h_B[i] = static_cast<double>(N - i);
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-6) {
            fprintf(stderr, "Mismatch at index %d: GPU %f != expected %f\n",
                    i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful. All results match.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```