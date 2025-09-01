/*
The formula for bandwidth is `(bytes / 1e9) / (milliseconds / 1000.0)`.

In order to demonstrate this formula in a CUDA program, we need to
measure how many bytes are transferred over the PCIe bus in a given
time interval.  The most straightforward way is to copy a large
buffer from host memory to device memory (or vice versa) and use
CUDA events to time the copy.  Once we have the elapsed time in
milliseconds and the amount of data in bytes, we can plug these
values into the given formula to obtain the bandwidth in gigabytes
per second.

The program below does exactly that:

1. Allocate a large array on the host and fill it with dummy data.
2. Allocate a corresponding array on the device.
3. Use `cudaMemcpy` (or `cudaMemcpyAsync` with events) to transfer
   the data from host to device, timing the operation with
   `cudaEventRecord`.
4. Compute the elapsed time in milliseconds and the amount of data
   transferred in bytes.
5. Apply the formula to compute the bandwidth in GB/s.
6. Print the result.

The code also includes a simple kernel that runs on the device
to demonstrate a computation, but the bandwidth calculation is
based solely on the memory transfer.  Error checking is performed
for all CUDA API calls.  This program is a self‑contained CUDA
C example that can be compiled with `nvcc` and executed on any
CUDA‑capable device.

Note: the formula `(bytes / 1e9) / (milliseconds / 1000.0)` yields
the bandwidth in gigabytes per second because:
- `bytes / 1e9` converts bytes to gigabytes,
- `milliseconds / 1000.0` converts milliseconds to seconds,
- dividing gigabytes by seconds gives GB/s.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                      \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",  \
                    #call, __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                      \
    } while (0)

// Simple kernel that adds 1 to each element
__global__ void increment_kernel(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] += 1;
    }
}

int main(void) {
    const size_t N_BYTES = 256 * 1024 * 1024; // 256 MB
    const size_t N_ELEMS = N_BYTES / sizeof(int);

    // Allocate host memory
    int *h_data = (int *)malloc(N_BYTES);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < N_ELEMS; ++i) {
        h_data[i] = 0;
    }

    // Allocate device memory
    int *d_data = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_data, N_BYTES));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N_BYTES, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N_ELEMS + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N_ELEMS);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N_BYTES, cudaMemcpyDeviceToHost));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // Compute bandwidth using the given formula
    double bandwidth_gb_s = (N_BYTES / 1e9) / (ms / 1000.0);

    printf("Transferred %zu bytes in %.3f ms\n", N_BYTES, ms);
    printf("Bandwidth: %.3f GB/s\n", bandwidth_gb_s);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    return 0;
}
