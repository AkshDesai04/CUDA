```cpp
/*
Aim: Allocate two separate device arrays. Try to use an index that is valid for the second array on a pointer to the first array.

Thinking Process:
- The goal is to demonstrate that using an index that is valid for one array on another array can lead to out‑of‑bounds memory access.
- Allocate two device arrays of different sizes (size1 < size2) using cudaMalloc.
- Write a simple kernel that takes both array pointers and their sizes.
- Inside the kernel, use the second array's size to launch threads.
- Each thread will write to the first array using its global index (idx).
- If idx is greater than or equal to size1, the write will be out‑of‑bounds on the first array.
- After kernel launch, copy back the first array and print its contents.
- Running this program under cuda-memcheck will flag the out‑of‑bounds write.
- The code includes basic CUDA error checking and is self‑contained.

*/

#include <cstdio>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that writes to the first array using the second array's index range
__global__ void writeOutOfBounds(float* arr1, int size1, int size2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2)
    {
        // Intentionally using idx that may exceed size1
        arr1[idx] = static_cast<float>(idx);
    }
}

int main(void)
{
    const int size1 = 5;   // Size of the first array
    const int size2 = 10;  // Size of the second array (larger than size1)

    // Allocate host memory
    float h_arr1[size1];
    float h_arr2[size2];

    // Initialize host arrays
    for (int i = 0; i < size1; ++i) h_arr1[i] = 0.0f;
    for (int i = 0; i < size2; ++i) h_arr2[i] = static_cast<float>(i);

    // Allocate device memory
    float *d_arr1, *d_arr2;
    CHECK_CUDA(cudaMalloc((void**)&d_arr1, size1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_arr2, size2 * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_arr1, h_arr1, size1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_arr2, h_arr2, size2 * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel: one thread per element of the second array
    const int threadsPerBlock = 256;
    const int blocks = (size2 + threadsPerBlock - 1) / threadsPerBlock;
    writeOutOfBounds<<<blocks, threadsPerBlock>>>(d_arr1, size1, size2);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_arr1, d_arr1, size1 * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the first array
    printf("Resulting first array after kernel execution:\n");
    for (int i = 0; i < size1; ++i)
    {
        printf("h_arr1[%d] = %f\n", i, h_arr1[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_arr1));
    CHECK_CUDA(cudaFree(d_arr2));

    return 0;
}
```