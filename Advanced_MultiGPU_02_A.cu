/*
After the kernels complete, copy the results from both GPUs back to the host and assemble the final complete vector C.

Here is the reasoning and design of the program:

1. **Goal**  
   We want to demonstrate how to use multiple GPUs to perform a simple element‑wise operation on a large vector.  
   Each GPU will compute a portion of the result vector. After both kernels finish, we copy the partial results
   back to the host and combine them into the final vector `C`.

2. **Assumptions**  
   * The system has at least two CUDA‑capable GPUs.  
   * The vectors are of length `N` (here we use 1<<20 elements).  
   * The operation is `C[i] = A[i] + B[i]`.  

3. **Steps**  
   - Allocate host arrays `A`, `B`, `C`.  
   - Fill `A` and `B` with sample data.  
   - For each GPU (`gpu_id = 0` and `1`):  
     * Set the device (`cudaSetDevice`).  
     * Compute the start index and length for this GPU.  
     * Allocate device memory for its segment of `A`, `B`, and `C`.  
     * Copy the relevant slice of `A` and `B` to device.  
     * Launch the addition kernel.  
     * Copy the result slice of `C` back to the corresponding position in the host array.  
     * Free device memory.  
   - After both GPUs finish, `C` holds the complete result.  
   - Optionally verify a few elements.

4. **Key CUDA API calls**  
   - `cudaSetDevice` to select the GPU.  
   - `cudaMalloc`, `cudaMemcpy` for memory management.  
   - `cudaMemcpyAsync` (with a stream) could be used for overlapping but we use simple `cudaMemcpy` for clarity.  
   - `cudaDeviceSynchronize` to ensure kernels finish before we copy results back.

5. **Error checking**  
   A helper macro `CHECK_CUDA` is defined to abort on any CUDA error.

6. **Compilation**  
   Compile with:  
   `nvcc -o multi_gpu_vector_add multi_gpu_vector_add.cu`  
   Run: `./multi_gpu_vector_add`

Now the full CUDA C program follows.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple element‑wise addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    // Size of the full vectors
    const int N = 1 << 20;          // 1,048,576 elements
    const int NUM_GPUS = 2;         // Using two GPUs

    // Allocate host memory
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 2.0f;
    }

    // Determine segment size for each GPU
    int segmentSize = N / NUM_GPUS;
    int remainder   = N % NUM_GPUS;

    // Process each GPU
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        // Set current device
        CHECK_CUDA(cudaSetDevice(gpu));

        // Compute offset and size for this GPU
        int offset = gpu * segmentSize;
        int size   = segmentSize;
        if (gpu == NUM_GPUS - 1) {
            // Last GPU gets the remainder
            size += remainder;
        }

        // Allocate device memory for this segment
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc((void**)&d_A, size * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_B, size * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_C, size * sizeof(float)));

        // Copy input segments to device
        CHECK_CUDA(cudaMemcpy(d_A, h_A + offset, size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B + offset, size * sizeof(float), cudaMemcpyHostToDevice));

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid  = (size + threadsPerBlock - 1) / threadsPerBlock;
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
        CHECK_CUDA(cudaGetLastError());

        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(h_C + offset, d_C, size * sizeof(float), cudaMemcpyDeviceToHost));

        // Free device memory
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }

    // Optionally verify results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            if (errors < 10) {
                printf("Mismatch at %d: got %f, expected %f\n", i, h_C[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("All results correct.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Clean up host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
