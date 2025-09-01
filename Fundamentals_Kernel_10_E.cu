```cuda
/*
Aim: Define a struct that contains a small, fixed-size array (e.g., float vec[3]). Pass it by value and have the kernel print the array elements.

Thinking:
- We need a simple struct that holds an array of three floats. This array is part of the struct layout, so passing the struct by value to a kernel should copy the entire array into registers or local memory.
- The kernel will accept this struct as a parameter and use the device-side printf to display each element. 
- For device printf to work we must compile for at least sm_20 (default with recent CUDA versions is fine). 
- In main() we create an instance of the struct, initialize it with sample data, and launch the kernel with a single thread block of one thread.
- After the kernel launch we synchronize the device and check for errors.
- The program is self‑contained and can be compiled with: nvcc -arch=sm_35 struct_pass.cu -o struct_pass
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Define the struct with a small fixed‑size array
struct SmallVec {
    float vec[3];
};

// CUDA kernel that receives the struct by value and prints its elements
__global__ void printVecKernel(SmallVec s)
{
    // Print each element
    printf("Thread %d: vec[0] = %f, vec[1] = %f, vec[2] = %f\n",
           threadIdx.x, s.vec[0], s.vec[1], s.vec[2]);
}

// Helper macro for error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main()
{
    // Create an instance of the struct on host and initialize
    SmallVec h_vec;
    h_vec.vec[0] = 1.0f;
    h_vec.vec[1] = 2.0f;
    h_vec.vec[2] = 3.0f;

    // Launch kernel with one block and one thread, passing the struct by value
    printVecKernel<<<1, 1>>>(h_vec);

    // Wait for the kernel to finish and check for errors
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
```