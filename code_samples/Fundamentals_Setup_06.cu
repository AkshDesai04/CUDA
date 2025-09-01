/*
Query and print the maximum dimensions (x, y, z) of a thread block for the primary CUDA device. The result should be displayed clearly, for example: 'Max Block Dimensions: [X, Y, Z]'.

Thought process:
1. The program is a simple CUDA host-only application; no device kernel is needed because we only query device properties.
2. We'll use cudaGetDeviceProperties to retrieve a cudaDeviceProp structure for device 0 (the primary device).
3. The maximum block dimensions are stored in prop.maxThreadsDim[0], [1], and [2] for x, y, and z respectively.
4. Error checking: if cudaGetDeviceProperties fails, print an error message and exit with non-zero status.
5. Output the dimensions in the required format: "Max Block Dimensions: [X, Y, Z]".
6. Include necessary headers: <cuda_runtime.h> and <cstdio>.
7. The program's main function will return 0 on success.

This meets the requirement: it queries the maximum thread block dimensions of the primary CUDA device and prints them clearly.
*/

#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int device = 0; // Primary CUDA device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    printf("Max Block Dimensions: [%d, %d, %d]\n",
           prop.maxThreadsDim[0],
           prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);

    return 0;
}
