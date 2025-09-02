/*
Aim of the program: Add error checking to the `cudaMemset` call.

Thinking:
The goal is to demonstrate how to add proper error checking around a cudaMemset operation in CUDA C.  
1. We include the necessary headers and define a helper macro `CHECK_CUDA` that calls `cudaGetLastError()` after a CUDA runtime API call and aborts if an error is returned.  
2. In `main()` we allocate a device array with `cudaMalloc`, then call `cudaMemset` to set all bytes to a specific value.  
3. Immediately after `cudaMemset` we invoke `CHECK_CUDA` to detect any error from that call.  
4. We copy the data back to host memory with `cudaMemcpy`, again checking for errors.  
5. Finally, we print the first few elements to confirm that `cudaMemset` worked correctly.  
6. All allocated device memory is freed before program exit.  
This simple example illustrates the pattern of error checking after each CUDA runtime call, especially for `cudaMemset`. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA runtime API call */
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void)
{
    const int N = 10;
    const size_t size = N * sizeof(int);

    /* Allocate device memory */
    int *dev_ptr = NULL;
    CHECK_CUDA(cudaMalloc((void **)&dev_ptr, size));

    /* Set device memory to a specific byte value (e.g., 0xFF) */
    CHECK_CUDA(cudaMemset(dev_ptr, 0xFF, size));

    /* Allocate host memory and copy data back from device */
    int *host_ptr = (int *)malloc(size);
    if (!host_ptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        cudaFree(dev_ptr);
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));

    /* Print the result */
    printf("Device memory after cudaMemset:\n");
    for (int i = 0; i < N; ++i) {
        printf("host_ptr[%d] = %d\n", i, host_ptr[i]);
    }

    /* Clean up */
    free(host_ptr);
    CHECK_CUDA(cudaFree(dev_ptr));

    return EXIT_SUCCESS;
}
