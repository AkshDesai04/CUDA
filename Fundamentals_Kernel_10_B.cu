```cuda
/* 
   Aim of the program (as requested in the prompt): 
   Modify the kernel to change a member of the struct it received. 
   Does this change affect the original struct on the host? Explain why or why not (pass-by-value).

   Thinking: 
   - Define a simple POD struct (plain old data) so it can be copied easily.
   - Allocate the struct on the host, set initial values, and print them.
   - Allocate a device copy of the struct and copy the host struct to device.
   - Launch a kernel that receives the struct *by value* (i.e., the argument is the struct itself, not a pointer).
     Inside the kernel, modify one member of the struct.  Because the struct is passed by value,
     the kernel receives a copy in registers or local memory; modifications affect only that copy.
   - After kernel execution, copy the device struct back to the host and print it.
     The host struct should remain unchanged, demonstrating that pass‑by‑value does not
     modify the original object on the host.
   - Add CUDA error checks and use printf in the kernel to observe the modified value.
   - The program is self‑contained and can be compiled with nvcc and run on any CUDA device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

typedef struct {
    int   a;
    float b;
} MyStruct;

/* Kernel that receives a struct by value and modifies one member */
__global__ void modifyStructKernel(MyStruct s) {
    // Modify the copied struct
    s.a = 999;
    s.b = 3.14159f;
    // Print from device (requires nvcc >= 7.0 for printf support)
    printf("Inside kernel: s.a = %d, s.b = %f\n", s.a, s.b);
}

int main(void) {
    // Host struct initialization
    MyStruct h_struct;
    h_struct.a = 10;
    h_struct.b = 2.71828f;

    printf("Host before kernel launch: h_struct.a = %d, h_struct.b = %f\n", h_struct.a, h_struct.b);

    // Allocate device memory for struct
    MyStruct *d_struct;
    cudaError_t err = cudaMalloc((void**)&d_struct, sizeof(MyStruct));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy host struct to device
    err = cudaMemcpy(d_struct, &h_struct, sizeof(MyStruct), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (Host->Device) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_struct);
        return 1;
    }

    // Launch kernel with one block, one thread
    modifyStructKernel<<<1, 1>>>(*d_struct);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_struct);
        return 1;
    }

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_struct);
        return 1;
    }

    // Copy device struct back to host
    err = cudaMemcpy(&h_struct, d_struct, sizeof(MyStruct), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (Device->Host) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_struct);
        return 1;
    }

    // Print host struct after kernel execution
    printf("Host after kernel execution: h_struct.a = %d, h_struct.b = %f\n", h_struct.a, h_struct.b);

    // Free device memory
    cudaFree(d_struct);

    return 0;
}
```
