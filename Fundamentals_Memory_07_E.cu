/*
Aim of the program:
Try to pass the address of a host variable (`int x; int* p = &x;`) to a kernel. Dereference it. Observe the error.

Thinking:
The goal is to demonstrate that a host memory address cannot be directly used by a device kernel.
We declare a host integer `x`, obtain its address `p`, and pass this pointer to a kernel that attempts
to dereference it. Since `p` refers to memory on the CPU side, the GPU will not have access to that
address, leading to an invalid device pointer error when the kernel executes.

The kernel writes the dereferenced value into a device array `d_out`. After the kernel launch, we check
`cudaGetLastError()` to capture the error and print it. If the launch succeeded (which it won't), we copy
the result back to host and print it. The program will compile and run, but will report an error such as
"invalid device pointer" because the kernel attempted to dereference a host pointer.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(int *p, int *out) {
    // Attempt to dereference a host pointer on the device
    out[0] = *p;
}

int main() {
    int x = 42;
    int *p = &x;   // Host pointer

    int *d_out = NULL;
    cudaMalloc((void**)&d_out, sizeof(int));

    // Launch kernel with a host pointer
    kernel<<<1, 1>>>(p, d_out);

    // Check for launch error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    } else {
        // If no error (unlikely), copy result back to host
        int out;
        cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Kernel succeeded, out = %d\n", out);
    }

    // Clean up
    cudaFree(d_out);
    return 0;
}
