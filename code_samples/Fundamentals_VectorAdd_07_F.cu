```cuda
/*
Aim of the program:
Handle large vectors with boundary checks.

Thinking:
The task is to write a CUDA program that can process very large one‑dimensional
vectors (arrays) safely.  The key safety measure is a bounds check inside the
kernel so that each thread only operates on elements that actually exist.
Without this check, if the grid size exceeds the vector length we would write
past the end of the memory, causing undefined behaviour or a crash.

Implementation details:

1. Host code
   - Allocate two large host vectors `h_input` and `h_output` of type float.
   - Initialize the input with a simple pattern (e.g. index * 1.0f).
   - Allocate corresponding device vectors `d_input` and `d_output`.
   - Copy input to the device.
   - Launch a kernel that multiplies each element by a scalar factor.
   - Copy the result back to the host.
   - Verify the result for a few elements.
   - Measure execution time with CUDA events.
   - Clean up all allocated memory.

2. Kernel
   - Compute a global index `idx` from block and thread indices.
   - Perform a boundary check `if (idx < N)` before writing to `d_output`.
   - Multiply the element by a scalar (stored in a constant memory variable
     for efficiency).

3. Error handling
   - Define a macro `CUDA_CHECK` that checks the return code of CUDA API
     calls and aborts with a useful message if an error occurs.

4. Grid/Block configuration
   - Use a block size of 256 threads (typical for many GPUs).
   - Compute the number of blocks needed: `blocks = (N + threadsPerBlock - 1) / threadsPerBlock`.
   - CUDA allows a large number of blocks in 1‑D, but to be safe we can
     limit to 65535 and fall back to a 2‑D grid if needed.  For simplicity
     we assume the vector size is small enough to fit within 1‑D limits.

5. Scalability
   - The program accepts an optional command‑line argument to set the vector
     length.  If omitted, a default size of 1 << 25 (≈33 million elements,
     ~128 MiB per array) is used.
   - Because the kernel contains a boundary check, the program can safely
     launch with a grid that covers more elements than the vector length,
     which is useful when experimenting with block/warp configurations.

6. Boundary checks
   - The only place where a thread could exceed the bounds is when computing
     `idx`.  The `if (idx < N)` guard ensures that any out‑of‑range thread
     simply does nothing.

With these design choices, the program is robust, portable, and demonstrates
how to safely process large vectors in CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API errors */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                     \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/* Constant memory for scalar multiplier (fast access) */
__constant__ float d_scale;

/* Kernel: multiply each element of input vector by scalar and write to output */
__global__ void multiplyVector(const float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * d_scale;
    }
}

int main(int argc, char *argv[])
{
    /* Default vector size: 33,554,432 elements (~128 MiB per array) */
    size_t N = 1 << 25;
    if (argc > 1) {
        N = strtoull(argv[1], NULL, 10);
        if (N == 0) {
            fprintf(stderr, "Invalid vector size specified.\n");
            return EXIT_FAILURE;
        }
    }

    printf("Vector size: %zu elements\n", N);
    size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_input  = (float *)malloc(bytes);
    float *h_output = (float *)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input vector */
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = (float)i;
    }

    /* Allocate device memory */
    float *d_input = NULL;
    float *d_output = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_input,  bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_output, bytes));

    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    /* Set scalar multiplier in constant memory */
    float h_scale = 2.5f;  /* Example scalar */
    CUDA_CHECK(cudaMemcpyToSymbol(d_scale, &h_scale, sizeof(float)));

    /* Kernel launch configuration */
    const int threadsPerBlock = 256;
    int blocks = (int)((N + threadsPerBlock - 1) / threadsPerBlock);
    if (blocks > 65535) {
        /* Fallback to 2D grid if needed */
        dim3 gridDim(65535, (blocks + 65535 - 1) / 65535);
        dim3 blockDim(threadsPerBlock, 1);
        /* Record events for timing */
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));

        multiplyVector<<<gridDim, blockDim>>>(d_input, d_output, N);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Kernel execution time: %f ms\n", ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        /* 1D grid */
        dim3 gridDim(blocks);
        dim3 blockDim(threadsPerBlock);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));

        multiplyVector<<<gridDim, blockDim>>>(d_input, d_output, N);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Kernel execution time: %f ms\n", ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    int ok = 1;
    for (int i = 0; i < 5; ++i) {
        float expected = h_input[i] * h_scale;
        if (fabsf(h_output[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_output[i], expected);
            ok = 0;
            break;
        }
    }
    if (ok) {
        printf("First 5 results verified: %f %f %f %f %f\n",
               h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);
    }

    /* Clean up */
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    printf("Done.\n");
    return 0;
}
```