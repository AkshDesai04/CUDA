```cuda
/*
Aim of the program:
After modifying the 2D array on the device with a kernel, copy it back to the host using `cudaMemcpy2D` and verify the results.

Thinking behind the implementation:
- We need a complete CUDA C example that demonstrates allocating a 2D array on the host,
  copying it to the device with a pitched allocation, modifying it with a kernel, copying it
  back, and verifying the contents.
- Using `cudaMallocPitch` and `cudaMemcpy2D` ensures that the code handles the extra
  padding that CUDA may introduce for alignment.  This is a common pattern when dealing
  with 2D data.
- The kernel will simply add 1 to each element; this is easy to reason about and verify.
- After copying back to the host we will print the array and compare against the expected
  values to confirm the operation succeeded.
- All CUDA API calls are wrapped with a simple error‑checking macro for clarity.
- The example uses a small 4x5 integer array for brevity; the same logic scales to any
  dimensions.

The program flow:
1. Allocate a host 2D array and initialize it.
2. Allocate pitched device memory and copy the host array to the device using `cudaMemcpy2D`.
3. Launch a kernel that increments each element by 1.
4. Copy the modified array back to the host with `cudaMemcpy2D`.
5. Verify that each element has been incremented and print the results.
6. Clean up device and host resources.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that adds 1 to each element of a 2D array
__global__ void addOneKernel(int *d_ptr, size_t pitch, int cols, int rows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x < cols && y < rows) {
        // Compute the address with the pitched offset
        int *rowPtr = (int*)((char*)d_ptr + y * pitch);
        rowPtr[x] += 1;
    }
}

int main(void)
{
    const int rows = 4;    // number of rows
    const int cols = 5;    // number of columns
    const size_t h_size = rows * cols * sizeof(int);

    // Allocate and initialize host array
    int *h_array = (int*)malloc(h_size);
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    printf("Host array before copy:\n");
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            h_array[r * cols + c] = r * cols + c; // simple pattern
            printf("%4d", h_array[r * cols + c]);
        }
        printf("\n");
    }

    // Device pointer and pitch
    int *d_ptr = NULL;
    size_t pitch = 0;

    // Allocate pitched memory on device
    CHECK_CUDA(cudaMallocPitch((void**)&d_ptr, &pitch, cols * sizeof(int), rows));

    // Copy host array to device (2D copy)
    CHECK_CUDA(cudaMemcpy2D(d_ptr, pitch,
                            h_array, cols * sizeof(int),
                            cols * sizeof(int), rows,
                            cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);
    addOneKernel<<<gridSize, blockSize>>>(d_ptr, pitch, cols, rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy2D(h_array, cols * sizeof(int),
                            d_ptr, pitch,
                            cols * sizeof(int), rows,
                            cudaMemcpyDeviceToHost));

    // Verify and print results
    printf("\nHost array after copy back:\n");
    int pass = 1;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int expected = r * cols + c + 1;
            int actual = h_array[r * cols + c];
            printf("%4d", actual);
            if (actual != expected) {
                pass = 0;
            }
        }
        printf("\n");
    }

    if (pass) {
        printf("\nVerification PASSED: All elements incremented correctly.\n");
    } else {
        printf("\nVerification FAILED: Unexpected values found.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_ptr));
    free(h_array);
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
```