```cuda
/*
Aim: Allocate memory, but forget to copy the pointer's value back from the device. Print the host variable. 
What does it show and why?

Thinking: In CUDA programming, device memory is allocated on the GPU side with cudaMalloc, which returns a pointer 
that is only valid on the device. The host pointer that receives this address does not automatically get updated 
unless we explicitly copy it back or assign it on the host side. If we mistakenly allocate memory on the device 
and then forget to copy the device pointer value back to a host variable that we intended to use later, the host 
variable will retain whatever value it had before the allocation (often NULL or some uninitialized value). 
Thus, when we print that host variable, it will show the original value, not the device pointer address, 
because the host side never received the new address. This demonstrates the importance of managing pointers 
correctly between host and device; device memory addresses cannot be dereferenced on the host, and host 
variables must be explicitly set to the device addresses if they are to be used for further kernel launches 
or data transfers.

This program allocates an integer array on the device, deliberately does NOT copy the device pointer back to 
the host variable that was intended to hold it, and then prints that host variable to show that it remains 
unchanged (NULL). It also prints the actual device pointer for comparison.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that does nothing, just to illustrate a kernel launch
__global__ void dummyKernel(int *d_data) {
    // No operation
}

int main(void) {
    int *h_ptr = NULL;          // Host pointer, initially NULL
    int *d_ptr = NULL;          // Device pointer

    // Allocate memory on the device
    size_t N = 10;
    size_t bytes = N * sizeof(int);
    cudaError_t err = cudaMalloc((void**)&d_ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Intentionally **do not** copy the device pointer back to h_ptr.
    // Normally we would do: h_ptr = d_ptr; or store d_ptr elsewhere.

    // Launch a dummy kernel (not really necessary for this demonstration)
    dummyKernel<<<1,1>>>(d_ptr);
    cudaDeviceSynchronize();

    // Print the host variable that was supposed to hold the device pointer
    printf("Host variable h_ptr after device allocation (not updated): %p\n", (void*)h_ptr);

    // For comparison, print the device pointer value
    printf("Device pointer d_ptr: %p\n", (void*)d_ptr);

    // Clean up
    cudaFree(d_ptr);
    return 0;
}
```