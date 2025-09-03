```c
/*
Aim: Define a nested struct (a struct that contains another struct as a member). Pass it by value to a kernel and access a member of the inner struct.

Thinking:
1. Create an `Inner` struct with a single float member `val`.
2. Create an `Outer` struct that contains an `int id` and an `Inner inner` member.
3. Define a CUDA kernel that receives an `Outer` struct by value. Inside the kernel, access `o.inner.val` and print it using `printf`. This demonstrates that the nested struct is correctly passed and its inner member is accessible.
4. In `main`, initialize an `Outer` instance on the host with sample data, launch the kernel with a simple launch configuration, and synchronize to ensure the output is printed.
5. Add minimal error handling for CUDA API calls to make the program robust.
6. The program uses only standard headers and CUDA runtime API, making it straightforward to compile with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Define a simple inner struct
struct Inner {
    float val;
};

// Define an outer struct that contains the inner struct
struct Outer {
    int  id;
    Inner inner;
};

// Kernel that receives an Outer struct by value and prints its inner member
__global__ void printOuter(Outer o)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d: Outer id=%d, inner.val=%f\n", idx, o.id, o.inner.val);
}

int main(void)
{
    // Initialize a sample Outer instance on the host
    Outer h_outer;
    h_outer.id = 42;
    h_outer.inner.val = 3.14159f;

    // Launch the kernel with 1 block of 8 threads
    printOuter<<<1, 8>>>(h_outer);

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}
```