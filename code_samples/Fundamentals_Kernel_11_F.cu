/*
Pass the struct pointer, but forget to copy the data from host to device first. What values does the kernel see when it reads from the struct's members?
In this program we define a simple struct `MyStruct` with an integer and a float member. We allocate an instance of this struct on the host and initialize its fields. Then we allocate space for the same struct on the device with `cudaMalloc`, but we deliberately omit the `cudaMemcpy` call that would normally transfer the initialized data from the host to the device. We launch a kernel that dereferences the device pointer to read the struct members and prints their values using `printf`. Because the device memory was never initialized with the host data, the values observed by the kernel are indeterminate: they are whatever happened to be in that region of device memory at the time of allocation (often garbage, but could occasionally be zero). If the device memory happened to be shared with other allocations or was previously used, the kernel may see remnants of those values. In any case, this demonstrates that forgetting to copy data to the device leads to undefined behavior and unreliable results.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Define a simple struct with two members
struct MyStruct {
    int   a;
    float b;
};

// Kernel that reads and prints the struct members
__global__ void readStructKernel(const MyStruct* d_ptr) {
    // Read the struct members
    int   val_a = d_ptr->a;
    float val_b = d_ptr->b;

    // Print the values (note: printf in device code prints to stdout)
    printf("Kernel sees: a = %d, b = %f\n", val_a, val_b);
}

int main(void) {
    // Host struct initialization
    MyStruct h_struct;
    h_struct.a = 42;
    h_struct.b = 3.14159f;

    // Device pointer declaration
    MyStruct* d_struct = nullptr;

    // Allocate memory on the device for the struct
    cudaError_t err = cudaMalloc((void**)&d_struct, sizeof(MyStruct));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Intentionally skip cudaMemcpy to transfer data from host to device
    // cudaMemcpy(d_struct, &h_struct, sizeof(MyStruct), cudaMemcpyHostToDevice);

    // Launch the kernel with a single thread
    readStructKernel<<<1, 1>>>(d_struct);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_struct);
        return -1;
    }

    // Clean up
    cudaFree(d_struct);

    // Inform user that program finished
    printf("Program finished.\n");

    return 0;
}
