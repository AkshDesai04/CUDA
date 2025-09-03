```cuda
/*
Aim: Instead of exit(-1), use cudaDeviceReset() before exiting to clean up the context.
Thinking: The user requests to modify a program so that before exiting due to an error, we reset the device to clean up. In CUDA, cudaDeviceReset() is the recommended way to reset device context before terminating. This will ensure that any pending operations are finished, and the device is left in a clean state. We will incorporate this in a simple CUDA program that demonstrates kernel launch and error handling. The code will call cudaDeviceReset() in both error and normal exit paths. The comment will contain the aim text verbatim, and we also explain the reasoning behind using cudaDeviceReset() and how the program implements it. 
*/

#include <stdio.h>
#include <stdlib.h>

#define N (1 << 20)   // 1M elements

__global__ void addKernel(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{
    float *h_a = NULL, *h_b = NULL, *h_c = NULL;
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    size_t size = N * sizeof(float);
    cudaError_t err;

    /* Allocate host memory */
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        /* Clean up host memory before exit */
        free(h_a); free(h_b); free(h_c);
        cudaDeviceReset();   // Clean up device context if any
        exit(-1);
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    err = cudaMalloc((void **)&d_a, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_a failed: %s\n", cudaGetErrorString(err));
        free(h_a); free(h_b); free(h_c);
        cudaDeviceReset();
        exit(-1);
    }

    err = cudaMalloc((void **)&d_b, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_b failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        free(h_a); free(h_b); free(h_c);
        cudaDeviceReset();
        exit(-1);
    }

    err = cudaMalloc((void **)&d_c, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_c failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b);
        free(h_a); free(h_b); free(h_c);
        cudaDeviceReset();
        exit(-1);
    }

    /* Copy data from host to device */
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to d_a failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        cudaDeviceReset();
        exit(-1);
    }

    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to d_b failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        cudaDeviceReset();
        exit(-1);
    }

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        cudaDeviceReset();
        exit(-1);
    }

    /* Wait for GPU to finish */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        cudaDeviceReset();
        exit(-1);
    }

    /* Copy result back to host */
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy from d_c failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        cudaDeviceReset();
        exit(-1);
    }

    /* Verify a few elements */
    for (int i = 0; i < 5; ++i) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    /* Clean up */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    /* Reset device before normal exit */
    cudaDeviceReset();

    return 0;
}
```