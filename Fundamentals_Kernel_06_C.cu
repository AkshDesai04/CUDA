```cuda
/*
Aim of the program:
Modify the kernel to increment the value at `data[0]`. The host should initialize the value to 100, copy it to the device, launch the kernel to increment it, copy it back, and verify the result is 101.

Thinking:
1. The program needs a simple kernel that performs an atomic or direct increment on the first element of an integer array.
2. The host will:
   - Declare an integer variable initialized to 100.
   - Allocate memory on the device for one integer.
   - Copy the host value to the device (Host-to-Device).
   - Launch the kernel with one block and one thread (since we only modify data[0]).
   - Synchronize to ensure the kernel completes.
   - Copy the updated value back to the host (Device-to-Host).
   - Verify the result is 101 and print success or failure.
3. Include basic error checking after each CUDA API call to aid debugging.
4. Keep the code minimal and self‑contained, compiling with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that increments data[0]
__global__ void inc(int *data) {
    data[0] += 1;
}

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main(void) {
    int h_data = 100;          // Host data initialized to 100
    int *d_data = NULL;        // Device pointer

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_data, sizeof(int)));

    // Copy from host to device
    CHECK_CUDA(cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel (1 block, 1 thread)
    inc<<<1, 1>>>(d_data);

    // Wait for kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    if (h_data == 101) {
        printf("Success: data[0] incremented to %d\n", h_data);
    } else {
        printf("Failure: expected 101, got %d\n", h_data);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
```