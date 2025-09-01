```cpp
/*
Write a program to allocate a 3D array using `cudaMalloc3D` and copy data to it using `cudaMemcpy3D`.

My approach is to:

1. Define the 3D dimensions (width, height, depth) and the element type (float).
2. Allocate a host array of size width*height*depth and fill it with sample data.
3. Create a CUDA pitched pointer for the host memory so that `cudaMemcpy3D` can understand the layout.
4. Use `cudaMalloc3D` to allocate device memory, obtaining a pitched pointer with device address, pitch, and extents.
5. Set up a `cudaMemcpy3DParms` structure: srcPtr, dstPtr, extent, and kind.
   - The extent width is in bytes (width * sizeof(float)).
   - srcPtr points to the host memory; dstPtr points to the device pitched pointer.
6. Call `cudaMemcpy3D` to copy the 3D block from host to device.
7. (Optional) Copy back to host using another `cudaMemcpy3D` call to verify the copy.
8. Print a few elements to confirm correctness.
9. Clean up all allocated memory.
10. Include error checking macros for CUDA API calls.

This will demonstrate allocation and 3D copy with `cudaMalloc3D` and `cudaMemcpy3D`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main()
{
    // 3D array dimensions
    const size_t width  = 4;   // X dimension (in elements)
    const size_t height = 4;   // Y dimension (in elements)
    const size_t depth  = 4;   // Z dimension (in elements)

    const size_t elemSize = sizeof(float);
    const size_t totalElems = width * height * depth;
    const size_t totalBytes = totalElems * elemSize;

    // Allocate and initialize host array (contiguous memory)
    float *h_array = (float*)malloc(totalBytes);
    if (!h_array)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Fill host array with some values
    for (size_t z = 0; z < depth; ++z)
    {
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                size_t idx = z * (height * width) + y * width + x;
                h_array[idx] = static_cast<float>(idx);
            }
        }
    }

    // Define the CUDA extent (width in bytes, height, depth)
    cudaExtent extent = make_cudaExtent(width * elemSize, height, depth);

    // Allocate device memory with cudaMalloc3D
    cudaPitchedPtr dPitched;
    CUDA_CHECK(cudaMalloc3D(&dPitched, extent));

    // Prepare source (host) pitched pointer
    cudaPitchedPtr hPitched = make_cudaPitchedPtr(
        (void*)h_array,          // devPtr points to host memory
        width * elemSize,        // pitch in bytes
        width,                   // xsize
        height);                 // ysize

    // Set up copy parameters
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = hPitched;
    copyParams.dstPtr = dPitched;
    copyParams.extent = extent;
    copyParams.kind   = cudaMemcpyHostToDevice;

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    // Optional: Copy back to host to verify
    float *h_array_back = (float*)malloc(totalBytes);
    if (!h_array_back)
    {
        fprintf(stderr, "Failed to allocate host memory for copy back\n");
        cudaFree(dPitched.devPtr);
        free(h_array);
        return EXIT_FAILURE;
    }

    cudaMemcpy3DParms copyParamsBack = {0};
    copyParamsBack.srcPtr = dPitched;
    copyParamsBack.dstPtr = make_cudaPitchedPtr(
        (void*)h_array_back,
        width * elemSize,
        width,
        height);
    copyParamsBack.extent = extent;
    copyParamsBack.kind   = cudaMemcpyDeviceToHost;

    CUDA_CHECK(cudaMemcpy3D(&copyParamsBack));

    // Print a few elements to verify
    printf("Sample elements after copy back:\n");
    for (size_t z = 0; z < depth; ++z)
    {
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                size_t idx = z * (height * width) + y * width + x;
                printf("%0.1f ", h_array_back[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Clean up
    free(h_array);
    free(h_array_back);
    CUDA_CHECK(cudaFree(dPitched.devPtr));

    return 0;
}
```