```cpp
/* 
Aim of the program: Allocate two separate device arrays. Try to use an index that is valid for the second array on a pointer to the first array.

Thinking:
1. We allocate two device arrays: d_arr1 with size N1 and d_arr2 with size N2.
2. We choose an index idx that is within the bounds of d_arr2 but out of bounds for d_arr1
   (e.g., N1 = 5, N2 = 10, idx = 9).
3. We launch a kernel that writes to d_arr1 using this out‑of‑bounds index.
4. This demonstrates undefined behaviour: the write may corrupt memory that belongs
   to d_arr2 or other device memory. Running the program under cuda-memcheck would
   flag the illegal memory access.
5. After the kernel we copy both arrays back to host and print their contents to
   show the potential corruption.

The code intentionally performs an out‑of‑bounds write to illustrate the issue.
Do not use such code in production; it is only for educational / debugging purposes.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

__global__ void write_out_of_bounds(float *arr, int idx, float val) {
    // Write to the array at the given index
    arr[idx] = val;
}

int main(void) {
    const int N1 = 5;   // Size of first array
    const int N2 = 10;  // Size of second array
    const int idx = N2 - 1;  // Index valid for d_arr2 but out of bounds for d_arr1

    float *d_arr1 = nullptr;
    float *d_arr2 = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_arr1, N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_arr2, N2 * sizeof(float)));

    // Initialize arrays on host
    float h_arr1[N1];
    float h_arr2[N2];
    for (int i = 0; i < N1; ++i) h_arr1[i] = 0.0f;
    for (int i = 0; i < N2; ++i) h_arr2[i] = 0.0f;

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_arr1, h_arr1, N1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_arr2, h_arr2, N2 * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel to write out of bounds on d_arr1
    write_out_of_bounds<<<1, 1>>>(d_arr1, idx, 42.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy arrays back to host
    CHECK_CUDA(cudaMemcpy(h_arr1, d_arr1, N1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_arr2, d_arr2, N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("Contents of d_arr1 after kernel:\n");
    for (int i = 0; i < N1; ++i) {
        printf("  h_arr1[%d] = %f\n", i, h_arr1[i]);
    }

    printf("\nContents of d_arr2 (should be unchanged, but may be corrupted):\n");
    for (int i = 0; i < N2; ++i) {
        printf("  h_arr2[%d] = %f\n", i, h_arr2[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_arr1));
    CHECK_CUDA(cudaFree(d_arr2));

    return 0;
}
```