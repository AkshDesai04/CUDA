```cpp
/* 
Test your modified kernel with non-multiple sizes like 500x500 and verify correctness. 
In order to test a CUDA kernel on matrix dimensions that are not multiples of the block size, 
the program below performs the following steps:

1. **Problem Choice** – An element‑wise addition of two matrices is chosen because it is 
   straightforward to verify against a CPU implementation and it requires only a simple 
   per‑thread computation.

2. **Kernel Design** – The kernel uses the classic 2‑D grid/block layout. Each thread 
   computes the sum of a single element. A bounds check is performed so that the last 
   block, which may partially cover the matrix, does not read/write out of bounds.

3. **Block and Grid Size** – A typical block size of 16×16 is chosen. To support sizes 
   such as 500×500, the grid dimensions are calculated using a ceiling division (`(N + blockSize - 1) / blockSize`).  
   This ensures that all elements are covered even when `N` is not divisible by the block size.

4. **Memory Management** – Host memory is allocated with `malloc`, initialized with 
   deterministic values (e.g. `A[i] = i`, `B[i] = 2*i`), and then copied to device 
   memory using `cudaMalloc` and `cudaMemcpy`. After kernel execution, the result is 
   copied back to host for verification.

5. **Verification** – The CPU performs the same addition in a simple loop and the 
   results are compared element‑by‑element. A tolerance is used to account for floating‑point 
   rounding. If all elements match, a success message is printed; otherwise an error is reported.

6. **Error Checking** – After each CUDA call and after kernel launch, `cudaGetLastError()` 
   is invoked to catch any runtime errors. The program exits with a non‑zero status if an error occurs.

7. **Performance Notes** – For this small test (500×500) the focus is correctness, not 
   throughput. The kernel is deliberately simple so that the correctness logic is clear. 
   In a real-world scenario, optimisations such as shared memory tiling or loop unrolling 
   could be applied and tested similarly.

The resulting .cu file below can be compiled with `nvcc` and executed to confirm that 
the kernel works correctly for non‑multiple block sizes.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

// Kernel to add two matrices element-wise
__global__ void addKernel(float *C, const float *A, const float *B, int N) {
    // Compute 2D indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * N + col;

    // Bounds check: only process elements that exist
    if (row < N && col < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Utility function to check CUDA errors
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    const int N = 500;          // Matrix dimension (not a multiple of block size)
    const int size = N * N;
    const int bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);   // Result from GPU
    float *h_ref = (float *)malloc(bytes); // Reference result from CPU

    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices
    for (int i = 0; i < size; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);
    checkCudaError("cudaMalloc");

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy HtoD");

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    addKernel<<<gridDim, blockDim>>>(d_C, d_A, d_B, N);
    checkCudaError("kernel launch");

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy DtoH");

    // Compute reference result on CPU
    for (int i = 0; i < size; ++i) {
        h_ref[i] = h_A[i] + h_B[i];
    }

    // Verify results
    int errors = 0;
    const float eps = 1e-5f;
    for (int i = 0; i < size; ++i) {
        if (fabs(h_C[i] - h_ref[i]) > eps) {
            errors++;
            if (errors <= 10) { // Print first few errors
                int row = i / N;
                int col = i % N;
                printf("Mismatch at (%d, %d): GPU=%f, CPU=%f\n",
                       row, col, h_C[i], h_ref[i]);
            }
        }
    }

    if (errors == 0) {
        printf("All %d elements match. Test PASSED.\n", size);
    } else {
        printf("Total mismatches: %d. Test FAILED.\n", errors);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```