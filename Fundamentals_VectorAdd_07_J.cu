/*
Aim: Time the squaring kernel.

Thinking:
- The goal is to write a CUDA C program that allocates an array on the host, copies it to the device, launches a kernel that squares each element, then copies the result back. 
- The execution time of the kernel itself will be measured using CUDA events (cudaEventRecord, cudaEventElapsedTime). 
- We’ll use a simple float array of size 1<<20 (1,048,576 elements) for demonstration. 
- The kernel will be launched with 256 threads per block and a grid size calculated to cover the whole array. 
- Error checking is important, so a macro will be used to check CUDA API calls. 
- The program will print the time taken for the kernel launch in milliseconds and optionally display a few elements of the resulting array for verification. 
- No external libraries beyond the CUDA runtime and standard C headers will be used, ensuring the file can be compiled with nvcc as a standalone .cu file. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA API calls */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel that squares each element of the array */
__global__ void square_kernel(float *a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        a[idx] = val * val;
    }
}

int main(void)
{
    const int N = 1 << 20;          /* 1M elements */
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(size);
    if (h_a == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
    }

    /* Allocate device memory */
    float *d_a = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, size));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    /* Setup execution configuration */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Record the start event */
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* Launch kernel */
    square_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, N);

    /* Record the stop event */
    CUDA_CHECK(cudaEventRecord(stop, 0));

    /* Wait for the stop event to complete */
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Calculate elapsed time */
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    /* Clean up events */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost));

    /* Print timing result */
    printf("Kernel execution time: %f ms\n", milliseconds);

    /* Optionally verify a few results */
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_a[%d] = %f\n", i, h_a[i]);
    }

    /* Free device and host memory */
    CUDA_CHECK(cudaFree(d_a));
    free(h_a);

    /* Reset device */
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
