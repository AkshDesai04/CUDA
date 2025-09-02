```cu
/*
Aim of the program: Make the macro a `do { ... } while(0)` block to ensure it behaves like a single statement.

Thinking:
- The goal is to demonstrate how to define a macro that expands to a block of code but behaves syntactically like a single statement. 
- In CUDA C, this is useful when the macro is used in control structures such as `if` or loops; wrapping the macro body in `do { ... } while(0)` prevents issues with dangling else or missing braces.
- I will create a simple macro that prints an integer value from both host and device code. 
- The macro will be used in a device kernel to print each element of an array. 
- The kernel will launch enough threads to cover the array. 
- The host code will set up data, copy it to the device, launch the kernel, and then clean up. 
- Error checking is omitted for brevity but could be added if needed.
- This minimal example shows the macro definition, its use in device code, and a straightforward host program that runs on the GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro that behaves like a single statement using do { ... } while(0)
#define PRINT_DEVICE_INT(x) \
    do { \
        printf("Device Value: %d\n", (x)); \
    } while (0)

__global__ void printKernel(const int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Use the macro to print the element at this index
        PRINT_DEVICE_INT(d_arr[idx]);
    }
}

int main(void)
{
    const int N = 10;
    int h_arr[N];
    for (int i = 0; i < N; ++i)
        h_arr[i] = i * 10;

    int *d_arr = nullptr;
    cudaMalloc((void **)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with enough threads to cover all elements
    printKernel<<<1, N>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaFree(d_arr);
    return 0;
}
```