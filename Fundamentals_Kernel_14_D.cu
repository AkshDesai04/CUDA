/*
Use `cudaDeviceSynchronize()` after the kernel launch. This is a common way to catch asynchronous errors from the kernel execution itself, not just launch configuration errors.

In this example we perform a simple vector addition on the GPU. After launching the kernel we immediately call `cudaGetLastError()` to catch any launch‑time errors, then call `cudaDeviceSynchronize()` to wait for the kernel to finish. If the kernel itself fails (e.g., due to an illegal memory access), `cudaDeviceSynchronize()` will return a non‑zero error code which we can then report. This pattern is the standard way to ensure that any runtime errors inside the kernel are caught, because many errors are only detected when the device is synchronized with the host.

The code below demonstrates:
1. Memory allocation on host and device.
2. Data transfer between host and device.
3. Kernel launch with appropriate grid/block configuration.
4. Error checking after the launch and after `cudaDeviceSynchronize()`.
5. Cleanup of device memory.

Feel free to replace the kernel or the data sizes with your own requirements. The key point is the placement of `cudaDeviceSynchronize()` to catch asynchronous kernel errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple CUDA kernel that adds two vectors */
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/* Helper macro to check CUDA errors */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",           \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, size));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Determine grid and block dimensions */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    /* Check for launch errors */
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(launchErr));
        // Continue to attempt to synchronize to capture any runtime errors
    }

    /* Synchronize to catch any asynchronous errors */
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Kernel execution error (after sync): %s\n", cudaGetErrorString(syncErr));
        // Clean up and exit
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return EXIT_FAILURE;
    }

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify results (optional) */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at %d: expected %f, got %f\n",
                        i, expected, h_c[i]);
            }
            errors++;
        }
    }
    if (errors) {
        fprintf(stderr, "Vector addition produced %d errors.\n", errors);
    } else {
        printf("Vector addition succeeded, no errors detected.\n");
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}
