```cuda
/*
Allocate memory with cudaMallocManaged. Now try to access this pointer from the host. Does it work? Why?

Thinking:
- cudaMallocManaged allocates unified memory accessible from both the host and the device.
- When accessed on the host, the pointer is valid; the runtime will migrate the page if it is not resident in the host memory.
- However, if the device writes to that memory, the host must call cudaDeviceSynchronize() or cudaMemcpy to ensure data is visible.
- The code below demonstrates allocation, a kernel that writes to the unified memory, synchronizing, and then reading from the host.
- The program prints the results, showing that host access works.
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void writeKernel(int *data, int val) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Simple write: each thread writes its index plus val
    data[idx] = idx + val;
}

int main(void) {
    const int N = 10;
    int *unifiedArray;

    // Allocate unified memory
    cudaError_t err = cudaMallocManaged(&unifiedArray, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch kernel to write values
    writeKernel<<<1, N>>>(unifiedArray, 100);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(unifiedArray);
        return -1;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(unifiedArray);
        return -1;
    }

    // Access from host
    printf("Unified memory contents accessed from host:\n");
    for (int i = 0; i < N; ++i) {
        printf("unifiedArray[%d] = %d\n", i, unifiedArray[i]);
    }

    // Clean up
    cudaFree(unifiedArray);
    return 0;
}
```