```cpp
/*
Aim of the program:
Allocate memory, copy a value to it, but forget to launch the kernel. Copy the value back. Does it match?

Thinking:
- We want to demonstrate that if a kernel is not launched, the data on the device remains unchanged.
- Allocate device memory using cudaMalloc, copy a known value from host to device.
- Skip launching any kernel (so no modifications on the device).
- Copy the value back from device to host and compare it with the original value.
- Since nothing was done on the device, the copied value should be identical to the original.
- Use simple error checking to ensure each CUDA call succeeds.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            return EXIT_FAILURE;                                           \
        }                                                                   \
    } while (0)

int main() {
    const int host_value = 42;
    int *device_ptr = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&device_ptr, sizeof(int)));

    // Copy value from host to device
    CHECK_CUDA(cudaMemcpy(device_ptr, &host_value, sizeof(int), cudaMemcpyHostToDevice));

    // Intentionally do NOT launch any kernel

    // Copy value back from device to host
    int host_value_back = 0;
    CHECK_CUDA(cudaMemcpy(&host_value_back, device_ptr, sizeof(int), cudaMemcpyDeviceToHost));

    // Compare and output result
    if (host_value_back == host_value) {
        printf("Success: Value matches. host_value = %d, host_value_back = %d\n",
               host_value, host_value_back);
    } else {
        printf("Error: Value does not match. host_value = %d, host_value_back = %d\n",
               host_value, host_value_back);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(device_ptr));

    return 0;
}
```