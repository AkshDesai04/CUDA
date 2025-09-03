```cuda
/*
Verify the results of SAXPY on the CPU.

The goal of this program is to compute the SAXPY operation, which is defined as
    y[i] = a * x[i] + y[i]
both on the GPU using a CUDA kernel and on the CPU using a simple loop.
After performing the operation on both devices, the program compares the
results element‑by‑element to ensure that the GPU computation is correct.
If any discrepancy is found, the program reports the index and the differing
values. This verification step is essential for debugging CUDA kernels and
ensuring numerical consistency between host and device computations.

Key steps:
1. Define a CUDA kernel that performs SAXPY on the device.
2. Allocate and initialize input arrays on the host.
3. Copy input data to the device.
4. Launch the kernel with enough blocks/threads to cover all elements.
5. Copy the GPU results back to the host.
6. Perform the same SAXPY operation on the CPU.
7. Compare the GPU and CPU results and report any mismatches.
8. Clean up all allocated memory.

The program uses simple error checking for CUDA API calls and prints a
success message if the GPU and CPU results match exactly. The array size and
scalar a are configurable via constants at the top of the file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1<<20)   // Number of elements (1 million)
#define A 2.5f      // Scalar multiplier
#define THREADS_PER_BLOCK 256

/* CUDA kernel for SAXPY: y = a * x + y */
__global__ void saxpy_kernel(const float *x, float *y, float a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        y[idx] = a * x[idx] + y[idx];
    }
}

/* CPU implementation of SAXPY */
void saxpy_cpu(const float *x, float *y, float a, int n)
{
    for (int i = 0; i < n; ++i)
    {
        y[i] = a * x[i] + y[i];
    }
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                              \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

int main()
{
    // Host arrays
    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_y_gpu = (float *)malloc(N * sizeof(float));
    float *h_y_cpu = (float *)malloc(N * sizeof(float));

    if (!h_x || !h_y_gpu || !h_y_cpu)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = (float)i * 0.001f;          // Example data
        h_y_gpu[i] = (float)i * 0.002f;      // Same initial y for GPU
        h_y_cpu[i] = h_y_gpu[i];             // Copy for CPU
    }

    // Device arrays
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void **)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_y, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y_gpu, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    saxpy_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_y, A, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU SAXPY
    saxpy_cpu(h_x, h_y_cpu, A, N);

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(h_y_gpu[i] - h_y_cpu[i]) > 1e-5f)
        {
            if (mismatches < 10) // Print first few mismatches
            {
                printf("Mismatch at index %d: GPU = %f, CPU = %f\n",
                       i, h_y_gpu[i], h_y_cpu[i]);
            }
            mismatches++;
        }
    }

    if (mismatches == 0)
    {
        printf("SAXPY verification succeeded: GPU and CPU results match.\n");
    }
    else
    {
        printf("SAXPY verification failed: %d mismatches found.\n", mismatches);
    }

    // Clean up
    free(h_x);
    free(h_y_gpu);
    free(h_y_cpu);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return EXIT_SUCCESS;
}
```