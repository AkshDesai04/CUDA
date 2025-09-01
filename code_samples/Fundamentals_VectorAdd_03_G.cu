```cuda
/*
Apply the boundary check to a kernel that takes two input vectors and one output vector.

This program demonstrates a simple element‑wise addition of two input vectors
(`a` and `b`) into an output vector (`c`).  The kernel uses a typical
boundary‑check pattern to ensure that each thread only operates on elements
that exist within the arrays.  The boundary check guards against out‑of‑bounds
accesses when the total number of threads launched (gridSize * blockSize)
exceeds the size of the data.

Key points considered in the implementation:

1. **Kernel signature** – The kernel receives pointers to the two input
   vectors and one output vector, along with the number of elements (`N`).
2. **Index calculation** – Each thread calculates its global linear index
   using `blockIdx.x`, `blockDim.x`, and `threadIdx.x`.
3. **Boundary check** – An `if (idx < N)` guard ensures that only valid
   indices perform the addition.  This is critical when the grid size is
   chosen to cover all elements but may leave a few threads without work.
4. **Host code** – Allocates host and device memory, initializes the
   input vectors with sample data, copies data to the device, launches the
   kernel, copies the result back, and verifies the output.
5. **Error handling** – A helper macro `CUDA_CHECK` wraps CUDA API calls
   to report errors and abort execution if something goes wrong.
6. **Performance note** – For large vectors, you can tune `blockSize` to
   match the GPU's warp size and occupancy considerations, but the
   boundary check remains essential regardless of block size.

This example is intentionally straightforward so that the boundary‑check
pattern is clear and can be adapted to more complex kernels that involve
multiple input and output arrays or multi‑dimensional data.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that adds two vectors with boundary check
__global__ void addVectors(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const int bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    bool ok = true;
    for (int i = 0; i < 10 && ok; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Mismatch at index %d: %f != %f + %f\n", i, h_c[i], h_a[i], h_b[i]);
            ok = false;
        }
    }
    if (ok) {
        printf("First 10 results verified successfully.\n");
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
```