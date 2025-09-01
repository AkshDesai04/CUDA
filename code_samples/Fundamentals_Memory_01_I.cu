/*
After allocating with `cudaMalloc`, use `cudaMemset` to set all the bytes of the device array to 0 before copying the host data over.
My thinking: I want to demonstrate the usage of cudaMalloc, cudaMemset, and cudaMemcpy in a simple CUDA program. The program will allocate a device array, zero it out using cudaMemset, then copy a host array into the device array. After that, it will copy back to host and print to verify the values. Error checking macros are used to catch CUDA API errors. No kernel launch is necessary for this demonstration. I will also keep the code straightforward and portable.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper macro for error checking */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

int main(void) {
    const int N = 10;
    int h_host[N];
    for (int i = 0; i < N; ++i) {
        h_host[i] = i + 1;    /* Sample host data: 1, 2, ..., N */
    }

    int *d_dev = NULL;
    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void **)&d_dev, N * sizeof(int)));

    /* Zero the device array before copying host data */
    CHECK_CUDA(cudaMemset(d_dev, 0, N * sizeof(int)));

    /* Copy host data over to device */
    CHECK_CUDA(cudaMemcpy(d_dev, h_host, N * sizeof(int), cudaMemcpyHostToDevice));

    /* Copy data back to host to verify */
    int h_result[N];
    CHECK_CUDA(cudaMemcpy(h_result, d_dev, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Device array contents after copy:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_result[i]);
    }
    printf("\n");

    /* Clean up */
    CHECK_CUDA(cudaFree(d_dev));

    return 0;
}
