/*
Aim: Run your leaky program with `cuda-memcheck --leak-check full` and analyze the output.

Thinking:
1. The goal is to create a CUDA program that deliberately leaks memory on both the host and the device.
2. We'll allocate several blocks of GPU memory using cudaMalloc without ever freeing them. 
3. Additionally, we will allocate host memory with malloc and never call free. 
4. To make the leaks more noticeable, we will perform a few iterations and allocate different sized blocks.
5. A dummy kernel launch is added to ensure the GPU memory is touched, forcing the CUDA runtime to manage the allocations and making the leaks visible to cuda-memcheck.
6. The program includes basic error checking for each allocation and kernel launch so that any failures are reported clearly.
7. When the program finishes, the allocated resources remain dangling. Running it under `cuda-memcheck --leak-check full` should report the device memory leaks and host memory leaks, giving the user a chance to analyze the output and understand which allocations were not cleaned up.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple kernel that does nothing but ensures device memory is accessed */
__global__ void dummyKernel(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] = data[idx] * 1.0f;   // trivial operation
}

int checkCudaStatus(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main()
{
    /* Allocate device memory in several chunks */
    float *d_buf1, *d_buf2, *d_buf3;
    size_t size1 = 256 * 1024 * sizeof(float);   // 256K floats
    size2 = 512 * 1024 * sizeof(float);          // 512K floats
    size3 = 1 * 1024 * 1024 * sizeof(float);     // 1M floats

    cudaMalloc((void **)&d_buf1, size1);
    if (checkCudaStatus("cudaMalloc d_buf1")) return 1;

    cudaMalloc((void **)&d_buf2, size2);
    if (checkCudaStatus("cudaMalloc d_buf2")) return 1;

    cudaMalloc((void **)&d_buf3, size3);
    if (checkCudaStatus("cudaMalloc d_buf3")) return 1;

    /* Allocate host memory that will not be freed */
    int *h_array = (int *)malloc(128 * sizeof(int));
    if (!h_array) {
        fprintf(stderr, "malloc h_array failed\n");
        return 1;
    }

    /* Launch dummy kernel to touch device memory */
    int threadsPerBlock = 256;
    int blocks = (512 + threadsPerBlock - 1) / threadsPerBlock;  // just enough to cover first buffer
    dummyKernel<<<blocks, threadsPerBlock>>>(d_buf1, 512);
    if (checkCudaStatus("kernel launch")) return 1;

    cudaDeviceSynchronize();
    if (checkCudaStatus("cudaDeviceSynchronize")) return 1;

    /* Intentionally omit all cudaFree and free calls */
    /* This is where the memory leaks occur. */

    /* Program exits here, leaving allocated memory unreleased. */
    return 0;
}
