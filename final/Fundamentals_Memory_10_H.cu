```c
/*
 * Aim of the program: Write a program to allocate a 3D array using `cudaMalloc3D` and copy data to it using `cudaMemcpy3D`.

 * Thinking and design:
 * 1. We need to allocate a 3D array on the device. CUDA provides cudaMalloc3D for this purpose. 
 *    It requires a cudaPitchedPtr structure that holds the device pointer, pitch (width in bytes of each row),
 *    and the allocation size in terms of width, height, and depth.
 *
 * 2. For simplicity we will use an array of floats. The array will have dimensions:
 *        width  = 8  (number of columns)
 *        height = 4  (number of rows)
 *        depth  = 3  (number of slices)
 *
 * 3. On the host we create a contiguous 3D array using a single 1D vector with size width*height*depth.
 *    We'll fill it with a simple pattern (e.g., value = z*100 + y*10 + x) so that we can verify after copying.
 *
 * 4. We allocate the device memory using cudaMalloc3D. The pitch returned by CUDA may be larger than width*sizeof(float)
 *    due to alignment requirements; we must keep track of it for correct host-to-device and device-to-host copies.
 *
 * 5. For the copy we use cudaMemcpy3D. This requires a cudaMemcpy3DParms struct. We'll set:
 *        srcPtr   = source host pointer wrapped in a cudaPitchedPtr
 *        srcPos   = (0,0,0)
 *        dstPtr   = destination device pointer (from cudaMalloc3D)
 *        dstPos   = (0,0,0)
 *        extent   = widthInBytes, height, depth
 *        kind     = cudaMemcpyHostToDevice
 *
 *    Note that srcPtr is a cudaPitchedPtr as well; we set its pitch to width*sizeof(float) and set memory type to cudaMemoryTypeHost.
 *
 * 6. After copying to the device, we copy the data back to a second host array to verify that the copy was successful.
 *
 * 7. We include basic error checking macros to simplify CUDA error handling.
 *
 * 8. The program prints the original host data and the data copied back from the device to verify correctness.
 *
 * 9. Finally we free all allocated memory (device and host).
 *
 * This program demonstrates the usage of cudaMalloc3D and cudaMemcpy3D with a simple 3D array of floats.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for error checking
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    const size_t width  = 8;   // columns
    const size_t height = 4;   // rows
    const size_t depth  = 3;   // slices

    const size_t totalElements = width * height * depth;
    const size_t bytes = totalElements * sizeof(float);

    // Host allocation
    float *h_src = (float *)malloc(bytes);
    if (!h_src) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t z = 0; z < depth; ++z) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = z * (height * width) + y * width + x;
                h_src[idx] = z * 100 + y * 10 + x; // simple pattern
            }
        }
    }

    // Device allocation with cudaMalloc3D
    cudaPitchedPtr devPitched;
    devPitched.pitch = 0; // will be set by cudaMalloc3D
    devPitched.ptr   = NULL;
    devPitched.xsize = width * sizeof(float);
    devPitched.ysize = height;
    devPitched.zsize = depth;

    CHECK_CUDA(cudaMalloc3D(&devPitched));

    // Prepare source pitched pointer for host-to-device copy
    cudaPitchedPtr srcPitched;
    srcPitched.pitch = width * sizeof(float); // host pitch
    srcPitched.ptr   = h_src;
    srcPitched.xsize = width * sizeof(float);
    srcPitched.ysize = height;
    srcPitched.zsize = depth;

    // Set up copy parameters
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = srcPitched;
    copyParams.dstPtr = devPitched;
    copyParams.extent = make_cudaExtent(width * sizeof(float), height, depth);
    copyParams.kind   = cudaMemcpyHostToDevice;

    // Perform the copy
    CHECK_CUDA(cudaMemcpy3D(&copyParams));

    // Allocate host buffer for verification
    float *h_dst = (float *)malloc(bytes);
    if (!h_dst) {
        fprintf(stderr, "Failed to allocate host memory for destination\n");
        CHECK_CUDA(cudaFree(devPitched.ptr));
        free(h_src);
        return EXIT_FAILURE;
    }

    // Prepare destination pitched pointer for device-to-host copy
    cudaPitchedPtr dstPitched;
    dstPitched.pitch = devPitched.pitch;
    dstPitched.ptr   = h_dst;
    dstPitched.xsize = width * sizeof(float);
    dstPitched.ysize = height;
    dstPitched.zsize = depth;

    // Set up copy parameters for device-to-host
    cudaMemcpy3DParms copyParamsBack = {0};
    copyParamsBack.srcPtr = devPitched;
    copyParamsBack.dstPtr = dstPitched;
    copyParamsBack.extent = make_cudaExtent(width * sizeof(float), height, depth);
    copyParamsBack.kind   = cudaMemcpyDeviceToHost;

    // Perform the copy back
    CHECK_CUDA(cudaMemcpy3D(&copyParamsBack));

    // Verify the data
    int mismatches = 0;
    for (size_t i = 0; i < totalElements; ++i) {
        if (h_src[i] != h_dst[i]) {
            fprintf(stderr, "Mismatch at index %zu: src=%f, dst=%f\n", i, h_src[i], h_dst[i]);
            mismatches++;
            if (mismatches > 10) break;
        }
    }

    if (mismatches == 0) {
        printf("Data copied correctly! No mismatches found.\n");
    } else {
        printf("Found %d mismatches.\n", mismatches);
    }

    // Cleanup
    CHECK_CUDA(cudaFree(devPitched.ptr));
    free(h_src);
    free(h_dst);

    return EXIT_SUCCESS;
}
```