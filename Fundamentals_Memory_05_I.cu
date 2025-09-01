```cpp
/*
Aim of the program:
Allocate memory, launch a kernel that uses it, synchronize, and then free the memory.

Thinking:
The requirement is to demonstrate the basic workflow of CUDA memory management and kernel execution in a single, self‑contained program.  
The program will:

1. Allocate a small array on the GPU using cudaMalloc.
2. Launch a very simple kernel that increments each element of the array.
3. Synchronize the device to ensure the kernel has finished.
4. Copy the resulting data back to the host using cudaMemcpy.
5. Print the results to verify that the kernel executed correctly.
6. Free the GPU memory with cudaFree and also clean up the host memory.

To keep the example minimal yet complete, the kernel will run with one thread per array element, and we’ll use a 1‑D block and grid configuration.  
Error checking macros are included for robustness, but the code focuses on the essential steps: allocation, kernel launch, synchronization, and deallocation.

*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple kernel that adds 1 to each element of the array */
__global__ void add_one_kernel(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] += 1;
    }
}

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void) {
    const int N = 10;                // Size of the array
    const size_t size = N * sizeof(int);

    int *h_arr = (int*)malloc(size); // Host array
    int *d_arr;                     // Device array

    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&d_arr, size));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));

    /* Determine block and grid sizes */
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    add_one_kernel<<<blocks, threadsPerBlock>>>(d_arr, N);
    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Synchronize to ensure kernel completion */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Result after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_arr));

    /* Free host memory */
    free(h_arr);

    /* Reset device (optional) */
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```