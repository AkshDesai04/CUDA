/*
Add a conditional compilation flag (e.g., `_DEBUG`) so the `CHECK` macro only performs the check in debug builds, and is empty in release builds for performance.

[Thinking]
The goal is to provide a CUDA example that demonstrates how to use a conditional compilation flag to enable or disable runtime checks based on the build configuration.  
In a typical development workflow, one might define a `_DEBUG` symbol for debug builds and leave it undefined for release builds.  
The `CHECK` macro should wrap a call to `cudaGetLastError()` (or any CUDA error-checking function) when `_DEBUG` is defined.  
In release mode the macro expands to nothing, thus incurring no runtime overhead.

The macro will be defined as:
```
#ifdef _DEBUG
#define CHECK(err) do { cudaError_t err_code = (err); if (err_code != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_code)); exit(EXIT_FAILURE); } } while(0)
#else
#define CHECK(err) ((void)0)
#endif
```
We will use this macro in a simple kernel launch to illustrate its usage.  The program will:
1. Allocate device memory.
2. Launch a trivial kernel that writes to the array.
3. Copy results back to host.
4. Print the results.
The `CHECK` macro will guard each CUDA API call when `_DEBUG` is defined, otherwise the calls are unchecked for speed.

Now we provide the complete .cu file implementing this behavior.  
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Conditional compilation of CHECK macro */
#ifdef _DEBUG
#define CHECK(err) do {                                          \
    cudaError_t err_code = (err);                               \
    if (err_code != cudaSuccess) {                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err_code)); \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)
#else
#define CHECK(err) ((void)0)
#endif

/* Simple CUDA kernel that fills an array with its index */
__global__ void fill_array(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = idx;
    }
}

int main(void) {
    const int N = 1024;
    int *d_arr = NULL;
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Allocate device memory */
    CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fill_array<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    /* Check for kernel launch errors */
    CHECK(cudaGetLastError());

    /* Copy result back to host */
    CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print first 10 results */
    printf("First 10 elements of the array:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    /* Clean up */
    CHECK(cudaFree(d_arr));
    free(h_arr);

    return EXIT_SUCCESS;
}
