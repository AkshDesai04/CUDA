/*
Aim of the program:
Write a function that takes an integer `N`, allocates an array of `N` floats on the device, and initializes all its elements to zero using `cudaMemset`. The function should return the device pointer.

Thought process:
1. We need to allocate device memory for `N` floats. The standard CUDA API function for allocation is `cudaMalloc`.
2. After allocation, we must set all bytes of the allocated memory to zero. The CUDA API function for this is `cudaMemset`. Since each element is a float (4 bytes), we pass the size in bytes as `N * sizeof(float)`.
3. `cudaMemset` expects an unsigned char value to set, so passing `0` works for zeroing floats as well.
4. The function should return the device pointer, i.e., a `float*` that points to the allocated memory.
5. Error checking is important: both `cudaMalloc` and `cudaMemset` can fail. We'll use a simple macro to check return codes and print an error message before exiting.
6. For demonstration, a `main` function will call this function, allocate an array of a few floats, copy it back to the host, print the values to confirm they are all zeros, and finally free the device memory.
7. The file will be a valid CUDA source file with a `.cu` extension, ready to compile with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Macro for checking CUDA errors following a CUDA API call */
#define CHECK_CUDA_ERROR(call)                                         \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in file '%s' in line %d : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/*
 * Function: allocateAndZero
 * -------------------------
 * Allocates an array of N floats on the device and initializes all elements to zero.
 *
 * N: Number of elements to allocate.
 *
 * returns: Pointer to the device memory containing the zero-initialized floats.
 */
float* allocateAndZero(int N) {
    float *d_ptr = NULL;
    size_t size = (size_t)N * sizeof(float);

    /* Allocate device memory */
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ptr, size));

    /* Initialize all bytes to zero */
    CHECK_CUDA_ERROR(cudaMemset(d_ptr, 0, size));

    return d_ptr;
}

/* Kernel to print device array (for demonstration purposes only) */
__global__ void printDeviceArray(const float *d_arr, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        printf("d_arr[%d] = %f\n", idx, d_arr[idx]);
    }
}

int main(void) {
    const int N = 10;
    float *d_arr = allocateAndZero(N);

    /* Optional: Launch a kernel to print the array from the device */
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    printDeviceArray<<<blocks, threadsPerBlock>>>(d_arr, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    /* Copy back to host to verify */
    float *h_arr = (float*)malloc(N * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Host array contents after cudaMemcpy:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_arr[%d] = %f\n", i, h_arr[i]);
    }

    /* Clean up */
    free(h_arr);
    CHECK_CUDA_ERROR(cudaFree(d_arr));

    return 0;
}
