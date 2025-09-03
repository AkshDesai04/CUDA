/*
After allocating with `cudaMalloc`, use `cudaMemset` to set all the bytes of the device array to 0 before copying the host data over.

Thinking:
- The goal is to demonstrate the correct order of CUDA memory operations.
- First, allocate a host array and initialize it with some data.
- Allocate device memory with `cudaMalloc`.
- Immediately zero the entire device buffer using `cudaMemset`. This is useful to ensure that any unused bytes are zeroed, which can be important for debugging or when the device memory will be used by a kernel that expects zeroed data.
- Then copy the host data to the device using `cudaMemcpy`.
- Finally, copy the data back to the host to verify that the copy succeeded and that the initial memset didn't affect the copied data.
- Include basic error checking for each CUDA call.
- Compile with: `nvcc memset_example.cu -o memset_example`
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const int N = 10;
    size_t size = N * sizeof(int);

    // Host allocation and initialization
    int *h_data = (int *)malloc(size);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_data[i] = i + 1; // simple data: 1,2,3,...
    }

    // Device allocation
    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, size));

    // Zero out the entire device array
    CHECK_CUDA(cudaMemset(d_data, 0, size));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Copy back to host to verify
    int *h_result = (int *)malloc(size);
    if (!h_result) {
        fprintf(stderr, "Failed to allocate host memory for result\n");
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Result from device:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_result[i]);
    }
    printf("\n");

    // Cleanup
    free(h_data);
    free(h_result);
    CHECK_CUDA(cudaFree(d_data));

    return EXIT_SUCCESS;
}
