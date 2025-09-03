```cpp
/*
Aim of the program:
Declare a global device variable using __device__. Write a kernel that modifies this variable. Write another kernel that reads it. Demonstrate how its value persists across kernel launches within the same application.

Thinking:
- A global device variable can be declared with the __device__ qualifier, e.g. __device__ int g_value. This variable is allocated in device memory and is visible to all kernels launched in the same application.
- Two simple kernels are needed:
  1. A setter kernel that writes a new value into the global variable.
  2. A reader kernel that prints the current value using device printf.
- To clearly demonstrate persistence across launches we will:
  1. Launch the setter kernel to set the variable to 10.
  2. Launch the reader kernel to print the value (should be 10).
  3. Launch the setter again with a different value (e.g., 20).
  4. Launch the reader again (should be 20).
- Device printf is used to display the value from within the kernel, providing a clear runtime trace.
- Additionally, we copy the variable back to host memory using cudaMemcpyFromSymbol to show that the host can observe the updated value after each launch.
- The kernels are launched with a single block and a single thread to avoid race conditions on the global variable.
- Basic CUDA error checking is performed via a macro to ensure any API failures are caught immediately.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Global device variable
__device__ int g_value = 0;

// Kernel that sets the global device variable
__global__ void set_value(int val) {
    // Only one thread performs the write to avoid race conditions
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_value = val;
    }
}

// Kernel that reads and prints the global device variable
__global__ void print_value(void) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Device read: g_value = %d\n", g_value);
    }
}

int main(void) {
    // Allocate host memory to copy the device variable back
    int h_value = 0;

    // Launch 1: Set value to 10
    set_value<<<1, 1>>>(10);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch 2: Read value
    print_value<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy device variable back to host and print
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_value, g_value, sizeof(int), 0, cudaMemcpyDeviceToHost));
    printf("Host read after first set: g_value = %d\n", h_value);

    // Launch 3: Set value to 20
    set_value<<<1, 1>>>(20);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch 4: Read value again
    print_value<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy device variable back to host again
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_value, g_value, sizeof(int), 0, cudaMemcpyDeviceToHost));
    printf("Host read after second set: g_value = %d\n", h_value);

    // Clean up
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```