/*
Write a kernel running on GPU 0 that takes a pointer to memory allocated on GPU 1 and accesses it directly (this requires Unified Virtual Addressing).

[Thinking]

The goal is to demonstrate that a CUDA kernel launched on one GPU (GPU 0) can directly access device memory that was allocated on another GPU (GPU 1). This requires Unified Virtual Addressing (UVA), which was introduced in CUDA 5.0 and allows the CPU and all GPUs to see a common address space. Even with UVA, GPUs need to enable peer access to read/write each other's memory without copying.

Approach:
1. Detect at least two GPUs. If fewer, abort.
2. Allocate an integer array on GPU 1 (`device1`).
3. Switch to GPU 0, enable peer access to GPU 1 (`cudaDeviceEnablePeerAccess`).
4. Launch a simple kernel on GPU 0 that writes its thread index into the array allocated on GPU 1.
5. Synchronize and then copy the array back from GPU 1 to the host for verification.
6. Print a few values to confirm the kernel ran successfully.
7. Clean up resources.

Error checking macros are added for clarity. The code is self‑contained and can be compiled with `nvcc` on a system that has at least two NVIDIA GPUs and CUDA Toolkit 5.0 or newer.

Compile with:
  nvcc -arch=sm_20 -o uva_example uva_example.cu

Run:
  ./uva_example

The program will print the first few elements of the array which should be 0, 1, 2, … confirming that GPU 0 wrote to GPU 1’s memory directly.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       
    } while (0)

__global__ void write_kernel(int *ptr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        ptr[idx] = idx;
    }
}

int main(void)
{
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2)
    {
        fprintf(stderr, "This demo requires at least two GPUs.\n");
        return EXIT_FAILURE;
    }

    const int device0 = 0;
    const int device1 = 1;
    const int N = 1024;              // Size of array

    int *d_arr_g1 = NULL;            // Pointer to memory on GPU 1

    /* Allocate memory on GPU 1 */
    CHECK_CUDA(cudaSetDevice(device1));
    CHECK_CUDA(cudaMalloc(&d_arr_g1, N * sizeof(int)));

    /* Enable peer access from GPU 0 to GPU 1 */
    CHECK_CUDA(cudaSetDevice(device0));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(device1, 0));

    /* Launch kernel on GPU 0 that writes into GPU 1's memory */
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    write_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr_g1, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back from GPU 1 to host for verification */
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr)
    {
        fprintf(stderr, "Host allocation failed.\n");
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device1));
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr_g1, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print first 10 elements to confirm correct operation */
    printf("First 10 elements written by GPU 0 into GPU 1's memory:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    /* Clean up */
    CHECK_CUDA(cudaSetDevice(device1));
    CHECK_CUDA(cudaFree(d_arr_g1));
    free(h_arr);

    return EXIT_SUCCESS;
}
