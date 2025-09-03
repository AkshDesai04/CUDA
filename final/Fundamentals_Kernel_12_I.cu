/*
 * Aim: Initialize a `dim3` variable with all three dimensions, e.g., `dim3 grid(4, 2, 2);`.
 *
 * Thinking:
 * 1. The goal is simply to show how to create a CUDA `dim3` object with specified
 *    x, y, and z dimensions.
 * 2. `dim3` is a helper type defined in <cuda_runtime.h>, so we include that header.
 * 3. In the host `main()` function we instantiate `dim3 grid(4, 2, 2);`.
 * 4. To verify the initialization, we print the values using std::cout.
 * 5. No device kernels or CUDA runtime calls are required for this minimal example.
 * 6. The program will compile with a standard CUDA compiler (nvcc) and run on the host.
 */

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    dim3 grid(4, 2, 2);
    std::cout << "Grid dimensions: (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n";
    return 0;
}