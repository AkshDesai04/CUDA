```c
/* 
Time the data transfer speed (bandwidth) of `cudaMemcpy`. Transfer a large array (e.g., 256MB) from host to device. Record the time before and after the copy using a CUDA event timer (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`). Calculate and print the bandwidth in GB/s.

My thinking:
- Allocate a large array on the host (256 MB). Using a `char` array keeps the size calculation straightforward.
- Allocate a device buffer of the same size with `cudaMalloc`.
- Create two CUDA events (`start` and `stop`) to measure the elapsed time of the copy operation.
- Record the start event, perform `cudaMemcpy` from host to device, record the stop event, then synchronize the stop event to ensure the copy has finished.
- Use `cudaEventElapsedTime` to obtain the elapsed time in milliseconds.
- Convert the elapsed time to seconds and calculate bandwidth as (size in bytes) / (elapsed time in seconds) / 1e9 to get GB/s.
- Print the bandwidth to standard output.
- Perform basic error checking after each CUDA API call for robustness.
- Clean up by freeing host and device memory and destroying events.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    const size_t size_bytes = 256UL * 1024 * 1024; // 256 MB
    char *h_ptr = NULL;
    char *d_ptr = NULL;
    cudaError_t err;

    // Allocate host memory
    err = cudaMallocHost((void**)&h_ptr, size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Host allocation failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Allocate device memory
    err = cudaMalloc((void**)&d_ptr, size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Device allocation failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_ptr);
        return -1;
    }

    // Create CUDA events
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate start failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate stop failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Record start event
    cudaEventRecord(start, 0);

    // Perform the copy from host to device
    err = cudaMemcpy(d_ptr, h_ptr, size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Record stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Compute bandwidth (GB/s)
    float elapsed_sec = elapsed_ms / 1000.0f;
    float bandwidth_gbps = (float)size_bytes / elapsed_sec / 1e9f;

    printf("Data transfer of %.0f MB took %.3f ms, bandwidth = %.3f GB/s\n",
           (float)size_bytes / (1024 * 1024), elapsed_ms, bandwidth_gbps);

cleanup:
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_ptr);

    // Free host memory
    cudaFreeHost(h_ptr);

    return 0;
}
```