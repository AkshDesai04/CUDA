```cuda
/*
Aim of the program:
What is the scope and lifetime of a __device__ variable? Explain in a comment. (Scope of the entire CUDA context, lifetime of the application).

Thinking and explanation:
- A __device__ variable declared at file scope (outside any function) has *global scope* on the device. 
  It is accessible from any kernel (and any device function) compiled for that module. 
- The compiler generates a symbol for this variable in the device code. 
- The variable resides in the device's global memory (or in the constant memory if declared __constant__), which is part of the CUDA context.
- The lifetime of the variable is tied to the CUDA context: it is created when the context is created (typically at the first CUDA API call that requires a context, or explicitly with cuCtxCreate) and it is destroyed when the context is destroyed (e.g., when the application exits or when cuCtxDestroy is called).
- Consequently, the variable persists across all kernel launches within the same context; its value is retained between launches.
- To observe this, we can declare a __device__ variable, modify it in a kernel, and copy its value back to the host after a launch. 
  Running multiple launches will show that the value is accumulated, confirming the persistent lifetime.
- Note: Device variables are not visible to the host directly; you must use cudaMemcpyFromSymbol (or its C++ API equivalents) to transfer data.
- If a __device__ variable is declared inside a function or as a static local, its scope remains limited to that function or translation unit, but its lifetime still follows the context.

The following code demonstrates these properties by using a global __device__ counter that is incremented by a kernel and then read back on the host after each launch.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Global __device__ variable with global scope on the device.
// It resides in the device's global memory and is valid for the lifetime of the CUDA context.
__device__ int device_counter = 0;

// Kernel that increments the device_counter.
__global__ void increment_counter(int increment) {
    // Each thread increments the counter by the specified amount.
    // For simplicity, we use only one thread here.
    device_counter += increment;
}

// Error checking macro
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                    \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

int main(void) {
    const int launches = 3;
    const int increments[launches] = {5, 10, 20};

    printf("Starting CUDA application. Device counter is initially 0.\n");

    for (int i = 0; i < launches; ++i) {
        // Launch the kernel with a single thread.
        increment_counter<<<1, 1>>>(increments[i]);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy the device counter back to host.
        int host_counter = 0;
        CHECK_CUDA(cudaMemcpyFromSymbol(&host_counter, device_counter, sizeof(int),
                                        0, cudaMemcpyDeviceToHost));

        printf("After launch %d (increment=%d), device_counter = %d\n",
               i + 1, increments[i], host_counter);
    }

    // Clean up and exit.
    printf("CUDA application finished.\n");
    return 0;
}
```