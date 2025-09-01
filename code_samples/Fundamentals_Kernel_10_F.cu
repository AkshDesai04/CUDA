/*
Aim of program: What happens if the struct contains a pointer member (e.g., `int* p`)? Pass it by value, and have the kernel try to dereference the pointer. Explain the result.

Thought process:
- In CUDA, device code runs on the GPU and can only access memory that resides in the device address space.
- If a struct on the host contains a pointer to host memory and that struct is copied to the device (via cudaMemcpy), the pointer value inside the struct is literally the host virtual address. 
- When the kernel receives this struct by value, it receives that host address, but the GPU does not have a mapping for it in its address space.
- Therefore, attempting to dereference the pointer inside the kernel will result in an invalid memory access. On most systems this manifests as a cudaErrorInvalidDevicePointer or a segmentation fault on the GPU side.
- The program below demonstrates this by:
  1. Creating an int array on the host.
  2. Initializing a struct containing a pointer to that array.
  3. Copying the struct to the device (only the pointer value is copied).
  4. Launching a kernel that prints the pointer value and tries to dereference it.
  5. Checking for errors after the kernel launch.
- The expected output is an error message indicating that the device tried to access an invalid pointer.
- To make the kernel safe, one would need to allocate memory on the device and copy the data there, then set the struct's pointer to point to that device memory before copying the struct to the device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

typedef struct {
    int *p;
} IntPtrStruct;

// Kernel that receives struct by value and attempts to dereference the pointer
__global__ void kernel(IntPtrStruct s) {
    // Print the pointer value (device printf requires device runtime)
    printf("Kernel sees pointer value: %p\n", (void*)s.p);
    // Attempt to dereference (will likely cause an invalid access)
    int val = *s.p; // This is the problematic line
    printf("Value at pointer: %d\n", val);
}

int main(void) {
    const int N = 5;
    int hostArray[N];
    for (int i = 0; i < N; ++i) hostArray[i] = i * 10;

    // Create struct on host pointing to hostArray
    IntPtrStruct hostStruct;
    hostStruct.p = hostArray;

    // Allocate device memory for the struct
    IntPtrStruct *d_struct;
    CUDA_CHECK(cudaMalloc((void**)&d_struct, sizeof(IntPtrStruct)));

    // Copy struct from host to device (only the pointer value is copied)
    CUDA_CHECK(cudaMemcpy(d_struct, &hostStruct, sizeof(IntPtrStruct), cudaMemcpyHostToDevice));

    // Launch kernel
    kernel<<<1,1>>>(*d_struct);

    // Synchronize and check for errors
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch or execution failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel executed without error (unexpected).\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_struct));

    return 0;
}
