```cuda
/*
Verify the results of subtraction on the CPU.

The aim of the program is to perform a subtraction operation on two
arrays of floating‑point numbers on the GPU, copy the result back to
the host, and then verify that the values match the results obtained
by performing the same subtraction on the CPU.  This serves as a
sanity check that the GPU kernel is producing correct results and
that the memory transfers between device and host are functioning
properly.

Thinking process:
1. Define a reasonable array size (e.g., 1<<20 elements) to provide
   a substantial workload.
2. Allocate host memory for two input arrays (h_a, h_b) and an output
   array (h_c_gpu) to receive the GPU result.  Allocate a CPU
   reference result array (h_c_cpu).
3. Initialize the input arrays with deterministic values
   (e.g., i and 2*i) so that we can easily compute the expected
   result on the CPU.
4. Allocate device memory for the same three arrays.
5. Copy input data from host to device.
6. Define a simple CUDA kernel that subtracts elements:
      c[i] = a[i] - b[i];
7. Launch the kernel with an appropriate grid/block configuration.
8. Copy the device output back to host (h_c_gpu).
9. On the host, compute the expected subtraction results into
   h_c_cpu.
10. Compare h_c_gpu and h_c_cpu element‑wise.  If any mismatch
    occurs, report an error; otherwise, report success.
11. Clean up all allocated memory.
12. Use a small helper macro for CUDA error checking to make the
    code robust and easier to read.

The program prints the total number of mismatches (zero if all
elements match) and exits with a status code reflecting success or
failure.  This fulfills the requirement of verifying GPU
subtraction results using the CPU as a reference.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/* Kernel performing element-wise subtraction: c = a - b */
__global__ void subtract_kernel(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] - b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c_gpu = (float *)malloc(bytes);   // Result from GPU
    float *h_c_cpu = (float *)malloc(bytes);   // Reference result

    if (!h_a || !h_b || !h_c_gpu || !h_c_cpu) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize input arrays with deterministic values */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
    }

    /* Allocate device memory */
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    subtract_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Compute reference result on CPU */
    for (int i = 0; i < N; ++i) {
        h_c_cpu[i] = h_a[i] - h_b[i];
    }

    /* Verify results */
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_c_gpu[i] - h_c_cpu[i]) > 1e-5f) {
            if (mismatches < 10) {  // Print first few mismatches
                printf("Mismatch at index %d: GPU %f vs CPU %f\n",
                       i, h_c_gpu[i], h_c_cpu[i]);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("All %d elements match. Verification succeeded.\n", N);
    } else {
        printf("Total mismatches: %d out of %d elements.\n", mismatches, N);
    }

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
```