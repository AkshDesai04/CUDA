```cpp
/*
 * The formula for bandwidth is `(bytes / 1e9) / (milliseconds / 1000.0)`.
 *
 * My thought process for creating this CUDA program is as follows:
 * 
 * 1. I want to demonstrate how to compute the bandwidth of a memory transfer
 *    (host to device or device to host) using CUDA.
 * 2. The program will:
 *    - Allocate a large array on the host.
 *    - Allocate a corresponding array on the device.
 *    - Use cudaMemcpy to transfer the data from host to device.
 *    - Measure the time taken for the transfer using CUDA events.
 *    - Compute the number of bytes transferred (size of array * element size).
 *    - Apply the given bandwidth formula `(bytes / 1e9) / (milliseconds / 1000.0)` 
 *      to obtain the bandwidth in GB/s.
 *    - Print the result.
 * 3. The program will be self‑contained, using only standard C/C++ headers and
 *    CUDA runtime API calls. It will compile with `nvcc`.
 * 4. Error checking will be performed on each CUDA API call to ensure any
 *    issues are caught and reported.
 * 5. The code will be written in a single `.cu` file as requested.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Utility macro for CUDA error checking */
#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    /* Define the number of elements and allocate host memory */
    const size_t num_elements = 1 << 28; /* ~268 million floats (~1GB) */
    const size_t bytes = num_elements * sizeof(float);

    float *h_data = (float *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host data */
    for (size_t i = 0; i < num_elements; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    /* Allocate device memory */
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_data, bytes));

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Record the start event */
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* Perform the host-to-device memory copy */
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    /* Record the stop event */
    CUDA_CHECK(cudaEventRecord(stop, 0));

    /* Wait for the event to complete */
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Compute elapsed time in milliseconds */
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    /* Compute bandwidth using the provided formula */
    double bandwidth = (bytes / 1e9) / (milliseconds / 1000.0); /* GB/s */

    /* Output the results */
    printf("Transferred %zu bytes in %.3f ms.\n", bytes, milliseconds);
    printf("Bandwidth: %.3f GB/s\n", bandwidth);

    /* Clean up */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
```