/*
Aim of the program (exact wording from prompt):
Pass a pointer to a struct to a kernel. The host should allocate the struct on both host and device. Initialize it on the host, copy to device. The kernel should modify a member of the struct. The host should copy it back and verify the modification.

Thinking process:
- Define a simple C++ struct (e.g., MyStruct) containing a couple of members.
- On the host, allocate an instance of this struct on the CPU, set its members to known values.
- Allocate device memory for the struct using cudaMalloc and get a device pointer.
- Copy the struct from host to device with cudaMemcpy.
- Launch a CUDA kernel that receives a pointer to MyStruct and modifies one member (e.g., increments an integer field).
- After kernel execution, copy the struct back from device to host.
- Verify the modification by printing the updated field and comparing with the expected value.
- Use basic error checking macros to simplify CUDA API error handling.
- The code is fully self‑contained in a single .cu file and can be compiled with nvcc.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Define a struct that will be used both on host and device
struct MyStruct {
    int  a;
    float b;
};

// Kernel that takes a pointer to MyStruct and modifies its members
__global__ void modifyStructKernel(MyStruct* d_struct) {
    // For demonstration, we just modify the fields
    if (d_struct) {
        d_struct->a += 1;          // Increment integer
        d_struct->b *= 2.0f;       // Double the float
    }
}

int main() {
    // Host allocation
    MyStruct h_struct;
    h_struct.a = 10;
    h_struct.b = 3.14f;

    printf("Host before copy: a = %d, b = %f\n", h_struct.a, h_struct.b);

    // Device allocation
    MyStruct* d_struct = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_struct, sizeof(MyStruct)));

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(d_struct, &h_struct, sizeof(MyStruct), cudaMemcpyHostToDevice));

    // Launch kernel: single thread is enough for this example
    modifyStructKernel<<<1, 1>>>(d_struct);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back from device to host
    CUDA_CHECK(cudaMemcpy(&h_struct, d_struct, sizeof(MyStruct), cudaMemcpyDeviceToHost));

    printf("Host after copy: a = %d, b = %f\n", h_struct.a, h_struct.b);

    // Verify modification
    if (h_struct.a == 11 && fabs(h_struct.b - 6.28f) < 1e-5f) {
        printf("Verification passed: struct was modified correctly.\n");
    } else {
        printf("Verification failed: unexpected struct values.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_struct));

    return 0;
}
