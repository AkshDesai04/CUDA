```cpp
/*
Aim of the program:
Pass an integer as an argument to the kernel and print its value from the device.

Thinking and design:

1. CUDA kernels can accept parameters just like normal C functions, but the parameters must
   be POD types (plain old data) and are passed by value. Here we want to pass an
   `int` value from host to device.

2. The device can print to stdout using `printf` only if the device supports compute
   capability 2.0 or higher. This is the simplest way to verify that the value
   arrives correctly.

3. The kernel will be very small – it just calls `printf` with the integer value.
   It does not need any thread indexing.

4. The host will:
   - define an `int` variable with some value,
   - launch the kernel with a single thread (`<<<1,1>>>`),
   - synchronize and check for errors.

5. Error checking is done with a helper function `checkCudaError` to keep the code
   clean.

6. The program is fully self‑contained and can be compiled with `nvcc`:

      nvcc -o print_int print_int.cu
   and then run:

      ./print_int

   You should see a line like `Value from device: 42` printed to the console.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define checkCudaError(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\"\n",
                file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

// Kernel that receives an integer and prints it
__global__ void printKernel(int val)
{
    // Only one thread will execute, so we can safely call printf
    printf("Value from device: %d\n", val);
}

int main(void)
{
    // Host integer to send to the device
    int hostValue = 42;

    // Launch kernel with one block of one thread
    printKernel<<<1, 1>>>(hostValue);

    // Wait for GPU to finish
    checkCudaError(cudaDeviceSynchronize());

    // Reset the device (optional but clean)
    checkCudaError(cudaDeviceReset());

    return 0;
}
```