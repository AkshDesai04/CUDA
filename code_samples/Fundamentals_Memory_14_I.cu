```cpp
/*
Aim: Pass a `nullptr` to the kernel and have the kernel try to read or write from it. Analyze the `cuda-memcheck` report for this null pointer dereference.

Thinking:
The goal is to demonstrate a null pointer dereference on the device and observe how cuda-memcheck reports it. 
We will:
1. Declare a device pointer `int *devPtr` initialized to `nullptr` (on the host).
2. Launch a simple kernel that receives this pointer and writes to the address it points to without any null check.  
   The kernel body will simply execute `*ptr = 123;`.  
   Because the pointer is null, this write will attempt to access the device memory address `0x0`, which is invalid.  
3. After the kernel launch we will call `cudaDeviceSynchronize()` and then check for errors with `cudaGetLastError()`.  
   In a normal execution this will not catch the invalid memory access because the error is generated inside the kernel; 
   however, running the program under `cuda-memcheck` will catch it and print a diagnostic message indicating a 
   “invalid device memory access at address 0x0”.  
4. The program will print any CUDA error codes so that when the user runs it normally they see no crash but can use
   `cuda-memcheck ./program` to see the detailed report.

No host memory allocation is performed; the program is intentionally minimal and only focuses on the null dereference.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple kernel that attempts to write to the memory location pointed to by ptr
__global__ void null_deref_kernel(int *ptr) {
    // Intentionally dereference without a null check
    *ptr = 123; // This will cause an invalid memory access when ptr is nullptr
}

int main() {
    // Host pointer initialized to nullptr
    int *devPtr = nullptr;

    // Launch the kernel with a single thread
    null_deref_kernel<<<1, 1>>>(devPtr);

    // Synchronize to catch any runtime errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize() returned error: %s\n", cudaGetErrorString(err));
    }

    // Check for errors from the kernel launch itself
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launched successfully (but may contain a null dereference).\n");
    }

    // Attempt to print a message to indicate completion
    printf("Program completed.\n");
    return 0;
}
```