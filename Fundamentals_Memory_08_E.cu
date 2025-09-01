/*
The formula for bandwidth is `(bytes / 1e9) / (milliseconds / 1000.0)`.

To demonstrate this formula in a CUDA context, we will:

1. Allocate a large array on the host and fill it with data.
2. Allocate a corresponding array on the device.
3. Copy data from host to device and back to host, measuring the elapsed time for each transfer using CUDA events.
4. Calculate the total number of bytes transferred (both directions) and the elapsed time in milliseconds.
5. Apply the provided bandwidth formula to compute the achieved bandwidth in gigabytes per second.
6. Print the result.

The program uses basic CUDA APIs (`cudaMalloc`, `cudaMemcpy`, `cudaEventRecord`, `cudaEventElapsedTime`) and standard C++ I/O. It is self‑contained and can be compiled with `nvcc`:

```bash
nvcc -o bandwidth bandwidth.cu
```

Running the compiled binary will display the measured bandwidth. */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main() {
    const size_t numElements = 1 << 24;          // 16M elements
    const size_t sizeInBytes = numElements * sizeof(float);

    // Allocate host memory
    float *h_data = (float*)malloc(sizeInBytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < numElements; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, sizeInBytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_data);
        return EXIT_FAILURE;
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Copy from host to device
    err = cudaMemcpy(d_data, h_data, sizeInBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Copy from device back to host
    err = cudaMemcpy(h_data, d_data, sizeInBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Total bytes transferred (H2D + D2H)
    size_t totalBytes = 2 * sizeInBytes;

    // Bandwidth calculation using the given formula
    // bandwidth_GBps = (bytes / 1e9) / (milliseconds / 1000.0)
    double bandwidthGBps = (static_cast<double>(totalBytes) / 1e9) / (static_cast<double>(milliseconds) / 1000.0);

    printf("Transferred %zu bytes in %.3f ms.\n", totalBytes, milliseconds);
    printf("Achieved bandwidth: %.3f GB/s\n", bandwidthGBps);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return EXIT_SUCCESS;
}
