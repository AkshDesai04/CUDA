/*
Aim: Try to pass the address of a host variable (`int x; int* p = &x;`) to a kernel. Dereference it. Observe the error.

Thinking:
1. In CUDA, device code runs on the GPU and can only directly access memory that belongs to the device (global, shared, or local memory).  
2. A host variable such as `int x` lives in CPU memory and its address (`&x`) is a *host* pointer.  
3. Passing this host pointer to a kernel and dereferencing it inside the kernel will cause the GPU to attempt to read from an address that is not mapped into the device’s address space.  
4. At runtime this manifests as an “invalid device pointer” error (or a segmentation fault, depending on the platform).  
5. The program below demonstrates this by:
   - Defining a host integer `x` and a pointer `p` that points to it.  
   - Launching a kernel that receives `p` and prints the value it points to.  
   - Checking for CUDA errors after the kernel launch and printing the error string.  
   - Observing that the program reports an invalid device pointer error when executed.  
6. This example is intentionally simple and does not attempt any memory copies; the goal is only to illustrate the failure of passing host pointers to device code.

Note: Some CUDA toolkits allow host pointers to be passed to kernels with a warning, but the actual dereference will still fail at runtime.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that attempts to dereference a pointer received from the host
__global__ void derefKernel(int* p) {
    // Attempt to read the value pointed to by the host pointer
    int val = *p;  // This should fail: p refers to host memory
    printf("Kernel read value: %d\n", val);
}

int main() {
    // Host variable
    int x = 42;
    int* p = &x;   // Host pointer to x

    printf("Host pointer address: %p\n", (void*)p);
    printf("Host variable value: %d\n", x);

    // Launch kernel with the host pointer (intentionally incorrect)
    derefKernel<<<1, 1>>>(p);

    // Synchronize to ensure kernel execution completes
    cudaError_t err = cudaDeviceSynchronize();

    // Check for errors
    if (err != cudaSuccess) {
        printf("Kernel launch or execution failed: %s\n", cudaGetErrorString(err));
    } else {
        // If no error reported by cudaDeviceSynchronize, still check for launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        } else {
            printf("Kernel executed successfully (unexpected).\n");
        }
    }

    // Clean up (not strictly necessary for this small program)
    cudaDeviceReset();

    return 0;
}
