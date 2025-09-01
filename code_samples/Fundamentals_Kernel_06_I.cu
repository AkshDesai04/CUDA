/*
Allocate memory, copy a value to it, but forget to launch the kernel. Copy the value back. Does it match?

Thinking:
The program is designed to demonstrate that if you allocate device memory, copy data from host to device, and then copy it back to host without launching any kernel to modify the data, the value retrieved should match the original value. This showcases that simply copying memory to and from the device without any computation preserves the data. The code will:
1. Allocate device memory for an integer.
2. Copy a host integer value to the device memory.
3. Intentionally skip kernel launch.
4. Copy the data back from the device to a new host variable.
5. Compare the original and retrieved values and print the result.
6. Clean up allocated device memory.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const int host_value = 42;
    int *dev_value = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&dev_value, sizeof(int)));

    // Copy value from host to device
    CHECK_CUDA(cudaMemcpy(dev_value, &host_value, sizeof(int), cudaMemcpyHostToDevice));

    // No kernel launch performed here

    // Copy value back from device to host
    int host_back = 0;
    CHECK_CUDA(cudaMemcpy(&host_back, dev_value, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify that the values match
    printf("Original host value: %d\n", host_value);
    printf("Value copied back from device: %d\n", host_back);
    printf("Values match? %s\n", (host_value == host_back) ? "Yes" : "No");

    // Clean up
    CHECK_CUDA(cudaFree(dev_value));

    return 0;
}
