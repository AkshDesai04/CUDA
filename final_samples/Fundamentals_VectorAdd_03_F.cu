```cuda
/*
 * Aim of the program:
 * Apply the same boundary check logic to a vector squaring kernel: `if (i < N) out[i] = in[i] * in[i]`.
 *
 * Thinking and design:
 *
 * 1. **Purpose**  
 *    Demonstrate a simple element‑wise operation (squaring) on a vector using CUDA.
 *    The kernel will apply a boundary check so that threads that map to indices
 *    beyond the vector length do nothing. This is the standard pattern for
 *    unrolled vector operations and protects against out‑of‑bounds memory
 *    accesses.
 *
 * 2. **Kernel design**  
 *    The kernel receives a pointer to the input array, a pointer to the output
 *    array, and the vector size `N`.  
 *    Each thread computes a global index `i = blockIdx.x * blockDim.x + threadIdx.x`.  
 *    If `i < N`, it writes `out[i] = in[i] * in[i]`.  
 *    The `if` guard ensures we never dereference past the end of the arrays.
 *
 * 3. **Host side**  
 *    - Allocate host memory for `h_in` and `h_out`.  
 *    - Initialize `h_in` with sample values (e.g., 0..N-1).  
 *    - Allocate device memory `d_in`, `d_out`.  
 *    - Copy `h_in` to `d_in`.  
 *    - Determine launch configuration:  
 *      Use 256 threads per block, compute the number of blocks as
 *      `(N + threadsPerBlock - 1) / threadsPerBlock`.  
 *    - Launch kernel.  
 *    - Copy results back to `h_out`.  
 *    - Verify or print a few elements.  
 *    - Clean up memory.  
 *
 * 4. **Error handling**  
 *    Wrap CUDA API calls with a simple macro to check return codes.  
 *
 * 5. **Build**  
 *    The file is a single `.cu` file; compile with:  
 *    `nvcc -o vec_square vec_square.cu`  
 *
 * 6. **Performance note**  
 *    This is a toy example; in real code one would use `cudaMemcpyAsync`,
 *    streams, and potentially device side reduction.  
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that squares each element of a vector with boundary check
__global__ void vector_square_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     const size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float val = in[i];
        out[i] = val * val;
    }
}

int main(void)
{
    const size_t N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in,  bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    // Launch kernel
    vector_square_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());          // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize());     // Ensure completion

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 elements
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_out[%d] = %f (expected %f)\n", i, h_out[i], h_in[i] * h_in[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);

    printf("Vector squaring completed successfully.\n");
    return EXIT_SUCCESS;
}
```