```c
/*
 * Aim of the program (as requested):
 * 
 * Try to pass the address of a host variable (`int x; int* p = &x;`) to a kernel. Dereference it. Observe the error.
 * 
 * Thinking behind the program:
 * 
 * 1. The goal is to demonstrate that CUDA device code cannot dereference a pointer that refers to a host address.
 * 2. In CUDA, pointers passed from host to kernel are interpreted as device addresses. If we pass a host address,
 *    the device will try to read from that address in device memory space, which is invalid.
 * 3. The program will allocate a host integer, take its address, and pass it to a simple kernel that tries to
 *    print the value at that pointer using device-side printf.
 * 4. After the kernel launch we will query `cudaGetLastError()` to capture the error code. We expect to see
 *    `cudaErrorInvalidDevicePointer` or a related message indicating that the kernel tried to access
 *    an invalid device address.
 * 5. To make the error visible, we also print the error string using `cudaGetErrorString`. This should
 *    confirm that passing a host address directly to a kernel leads to a runtime error.
 * 
 * The code below sets up the minimal infrastructure to demonstrate this:
 * - A kernel that dereferences the passed pointer and prints the value.
 * - Host code that passes the address of a local variable.
 * - Error checking after kernel launch.
 * 
 * When compiled with `nvcc` and run on a system with CUDA-capable GPU, the output should show a
 * CUDA error indicating an invalid device pointer. 
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that attempts to dereference the passed pointer */
__global__ void deref_kernel(int *p)
{
    /* Try to dereference the pointer and print the value.
     * This should fail because 'p' points to host memory. */
    printf("Kernel: *p = %d\n", *p);
}

int main(void)
{
    int x = 42;              /* Host integer */
    int *p = &x;             /* Host pointer to x */

    /* Launch the kernel with the host pointer */
    deref_kernel<<<1, 1>>>(p);

    /* Synchronize to catch the error */
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        /* Print the CUDA error that occurred during kernel execution */
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel executed successfully (unexpected).\n");
    }

    return 0;
}
```