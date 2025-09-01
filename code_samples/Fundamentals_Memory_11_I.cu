```cpp
/*
Aim of the program:
Try to use `cudaMemcpyFromSymbol` to read the value of the constant variable back to the host.

Thinking:
To demonstrate `cudaMemcpyFromSymbol`, we need a constant device variable that can be set from the host using `cudaMemcpyToSymbol`. The constant is defined with `__constant__`. After setting it, we will read its value back to the host with `cudaMemcpyFromSymbol`. The program will use error checking to ensure all CUDA calls succeed. We also print the retrieved value to confirm it matches what was written.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

// Define a constant device variable
__constant__ int const_val = 0;

// Simple kernel that does nothing but ensures device code is present
__global__ void dummyKernel() {
    // No operation
}

int main() {
    // Set a new value for the constant on the device
    int new_val = 7;
    CHECK_CUDA(cudaMemcpyToSymbol(const_val, &new_val, sizeof(int), 0, cudaMemcpyHostToDevice));

    // Optional: launch dummy kernel to ensure device initialization
    dummyKernel<<<1,1>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back the value from device constant to host
    int host_val = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&host_val, const_val, sizeof(int), 0, cudaMemcpyDeviceToHost));

    printf("Value of constant variable read from device: %d\n", host_val);

    // Clean up
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
```