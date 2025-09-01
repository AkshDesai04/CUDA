```cuda
/* 
Aim: Write a main function that tests a range of N values, some multiples of the block size and some not, and verifies all of them.

Thinking: 
- Choose a simple CUDA kernel, e.g., element-wise squaring of an array.
- Use a fixed block size (e.g., 256) and compute grid size accordingly: gridDim.x = (N + blockSize - 1)/blockSize.
- For each N in a list, allocate host input array, fill with test data, copy to device, run kernel, copy back result, compute reference on host, compare element-wise.
- Include error checking for CUDA API calls and kernel launches.
- Print pass/fail for each N.
- Ensure all memory is freed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/* Macro for checking CUDA errors */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* Kernel: square each element of the input array */
__global__ void squareKernel(const float *in, float *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = in[idx];
        out[idx] = val * val;
    }
}

int main(void)
{
    /* Test N values: some multiples of BLOCK_SIZE, some not */
    const int testNs[] = {1, 32, 63, 64, 65, 127, 128, 129,
                          256, 257, 512, 1023, 1024, 2048};
    const int numTests = sizeof(testNs) / sizeof(testNs[0]);

    for (int t = 0; t < numTests; ++t) {
        int N = testNs[t];
        size_t bytes = N * sizeof(float);

        /* Allocate host memory */
        float *h_in  = (float*)malloc(bytes);
        float *h_out = (float*)malloc(bytes);
        float *h_ref = (float*)malloc(bytes);
        if (!h_in || !h_out || !h_ref) {
            fprintf(stderr, "Host memory allocation failed for N=%d\n", N);
            exit(EXIT_FAILURE);
        }

        /* Initialize input and reference arrays */
        for (int i = 0; i < N; ++i) {
            h_in[i] = (float)i;
            h_ref[i] = h_in[i] * h_in[i];
        }

        /* Allocate device memory */
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc((void**)&d_in,  bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

        /* Copy input to device */
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        /* Launch kernel */
        int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        squareKernel<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());

        /* Copy result back */
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

        /* Verify results */
        bool success = true;
        for (int i = 0; i < N; ++i) {
            if (h_out[i] != h_ref[i]) {
                fprintf(stderr, "Mismatch at index %d: expected %f, got %f\n",
                        i, h_ref[i], h_out[i]);
                success = false;
                break;
            }
        }

        if (success)
            printf("Test passed for N = %d\n", N);
        else
            printf("Test failed for N = %d\n", N);

        /* Free memory */
        free(h_in);
        free(h_out);
        free(h_ref);
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    /* Reset device before exiting */
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```