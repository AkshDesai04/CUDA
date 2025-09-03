/*
Write a `main` function that calls your function, then copies the zeroed array back to the host and verifies that all elements are indeed 0.

The goal of this program is to demonstrate how to use a CUDA kernel to zero out an array on the GPU, copy the results back to the host, and validate that the operation succeeded.  
The program follows these steps:

1. **Kernel** – A simple CUDA kernel (`zeroArray`) that sets each element of a device array to 0.0f.  
2. **Helper Function** – A wrapper (`zeroArrayOnGPU`) that launches the kernel with appropriate grid/block configuration and synchronizes the device.  
3. **`main`** –  
   - Allocate a host array and a device array of a given size.  
   - Call `zeroArrayOnGPU` to zero the device array.  
   - Copy the device array back to the host using `cudaMemcpy`.  
   - Verify every element of the host array is zero and print the result.  
4. **Error Checking** – A macro (`CUDA_CHECK`) is used to catch and report any CUDA runtime errors immediately.

The program is self‑contained, uses standard CUDA APIs, and can be compiled with `nvcc`:

```
nvcc -o zero_array zero_array.cu
```

After running, it prints “All elements are zero.” if the operation succeeded. */

#include <stdio.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Kernel to zero out an array on the device */
__global__ void zeroArray(float* arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = 0.0f;
    }
}

/* Wrapper that launches the kernel and synchronizes the device */
void zeroArrayOnGPU(float* d_arr, int N)
{
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    zeroArray<<<blocks, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(void)
{
    const int N = 1 << 20;  // Example size: 1,048,576 elements

    /* Allocate host memory */
    float* h_arr = (float*)malloc(N * sizeof(float));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Allocate device memory */
    float* d_arr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(float)));

    /* Call the function that zeros the array on GPU */
    zeroArrayOnGPU(d_arr, N);

    /* Copy the zeroed array back to host */
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify all elements are zero */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != 0.0f) {
            ++errors;
            if (errors <= 10) {  // Print first few errors
                printf("Non-zero element at index %d: %f\n", i, h_arr[i]);
            }
        }
    }

    if (errors == 0) {
        printf("All elements are zero.\n");
    } else {
        printf("Found %d non-zero elements.\n", errors);
    }

    /* Clean up */
    free(h_arr);
    CUDA_CHECK(cudaFree(d_arr));

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
