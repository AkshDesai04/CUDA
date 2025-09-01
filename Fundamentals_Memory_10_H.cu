```cpp
/*
Aim of the program:
Write a program to allocate a 3D array using `cudaMalloc3D` and copy data to it using `cudaMemcpy3D`.

Thought process and design decisions:
1. The goal is to demonstrate the allocation of a 3D memory block on the device and transfer
   data from a host 3D array to that device memory.  We need to use the CUDA runtime
   API functions `cudaMalloc3D` and `cudaMemcpy3D`, which work with `cudaPitchedPtr` and
   `cudaMemcpy3DParms` structures.

2. We will choose modest dimensions for clarity: `dimX = 10`, `dimY = 20`, `dimZ = 30`.  
   Each element will be a `float`.  The host data will be allocated as a contiguous
   1D array of size `dimX*dimY*dimZ` and initialized with a simple pattern so that
   verification is easy.

3. For the device allocation we need a `cudaPitchedPtr` returned by `cudaMalloc3D`.  This
   structure contains a pointer to the device memory, the pitch (bytes per row), and
   the height (number of rows).  The width in bytes is computed as `dimX * sizeof(float)`.

4. The copy operation is performed with `cudaMemcpy3D`.  We set up a `cudaMemcpy3DParms`
   structure:
   - `srcPtr` and `dstPtr` are of type `cudaPitchedPtr`.  `srcPtr` points to the host
     memory and its pitch is `dimX * sizeof(float)`.  `dstPtr` comes from the result
     of `cudaMalloc3D`.
   - `extent` specifies the width, height, and depth of the region to copy.
   - `kind` is `cudaMemcpyHostToDevice`.

5. To verify the copy, we will copy the data back to a second host array using the same
   `cudaMemcpy3D` but with `kind = cudaMemcpyDeviceToHost` and then print a few elements.

6. Error checking is essential.  We implement a helper macro `CUDA_CHECK` that checks the
   return value of each CUDA call and prints an error message with the line number if it
   fails.

7. After the copy and verification, we free the device memory with `cudaFree3D` and
   deallocate the host arrays.

8. The program is written as a single `.cu` file that can be compiled with `nvcc`.

The final code implements the above steps and includes detailed comments for clarity.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),        \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void)
{
    // Dimensions of the 3D array
    const size_t dimX = 10;
    const size_t dimY = 20;
    const size_t dimZ = 30;
    const size_t elementSize = sizeof(float);

    // Allocate and initialize host 3D data (as 1D contiguous array)
    size_t hostSize = dimX * dimY * dimZ;
    float* h_data = (float*)malloc(hostSize * elementSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < hostSize; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Device 3D allocation
    cudaPitchedPtr d3dPtr;
    CUDA_CHECK(cudaMalloc3D(&d3dPtr, make_cudaExtent(dimX * elementSize, dimY, dimZ)));

    // Setup source (host) pitched pointer
    cudaPitchedPtr h3dPtr;
    h3dPtr.ptr = h_data;
    h3dPtr.pitch = dimX * elementSize;
    h3dPtr.xsize = dimX * elementSize;
    h3dPtr.ysize = dimY;

    // Setup copy parameters: Host -> Device
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = h3dPtr;
    copyParams.dstPtr = d3dPtr;
    copyParams.extent = make_cudaExtent(dimX * elementSize, dimY, dimZ);
    copyParams.kind = cudaMemcpyHostToDevice;

    // Perform copy
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    // Verification: copy back to a second host array
    float* h_verify = (float*)malloc(hostSize * elementSize);
    if (!h_verify) {
        fprintf(stderr, "Failed to allocate host verification memory.\n");
        CUDA_CHECK(cudaFree3D(d3dPtr));
        free(h_data);
        return EXIT_FAILURE;
    }

    // Setup destination (host) pitched pointer for verification
    cudaPitchedPtr h3dPtrVerify;
    h3dPtrVerify.ptr = h_verify;
    h3dPtrVerify.pitch = dimX * elementSize;
    h3dPtrVerify.xsize = dimX * elementSize;
    h3dPtrVerify.ysize = dimY;

    // Setup copy parameters: Device -> Host
    copyParams.srcPtr = d3dPtr;
    copyParams.dstPtr = h3dPtrVerify;
    copyParams.extent = make_cudaExtent(dimX * elementSize, dimY, dimZ);
    copyParams.kind = cudaMemcpyDeviceToHost;

    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    // Print a few elements to confirm correctness
    printf("Sample values after copy back from device:\n");
    for (size_t z = 0; z < 3; ++z) {
        for (size_t y = 0; y < 3; ++y) {
            for (size_t x = 0; x < 3; ++x) {
                size_t idx = z * (dimY * dimX) + y * dimX + x;
                printf("%f ", h_verify[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree3D(d3dPtr));
    free(h_data);
    free(h_verify);

    printf("3D array allocation and copy completed successfully.\n");
    return EXIT_SUCCESS;
}
```