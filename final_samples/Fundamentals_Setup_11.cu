/*
Aim: Write a simple __global__ function (a kernel) that does nothing. In your main function, attempt to compile this file using the NVIDIA CUDA Compiler (NVCC) command line, for example: nvcc my_program.cu -o my_program. Run the resulting executable.

Thinking:
- The kernel should be declared with __global__ and perform no operations; an empty body is sufficient.
- The main function should compile normally with nvcc and launch the kernel.
- To demonstrate that the program runs, we print messages before and after the kernel launch.
- Use cudaDeviceSynchronize() to ensure the kernel has finished executing before printing the final message.
- Basic error checking is added to report any launch or synchronization errors.
- The code is self‑contained and can be compiled with a command like:
      nvcc my_program.cu -o my_program
  Then run the executable:
      ./my_program
*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // This kernel intentionally does nothing.
}

int main()
{
    std::cout << "Launching dummy kernel..." << std::endl;

    // Launch the kernel with a single block of a single thread.
    dummyKernel<<<1, 1>>>();

    // Check for any launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Wait for the device to finish the kernel.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "cudaDeviceSynchronize returned error: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Dummy kernel executed successfully." << std::endl;
    return EXIT_SUCCESS;
}
