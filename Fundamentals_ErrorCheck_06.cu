/*
Create a C++ helper function `void checkKernelLaunch()` that calls `cudaDeviceSynchronize()` and checks for errors. Call this function after every kernel launch in your programs to ensure the kernel executed without an asynchronous error.

I started by parsing the requirement: we need a small CUDA program that contains a helper function named `checkKernelLaunch()` which will perform two tasks after a kernel launch:

1. Call `cudaGetLastError()` to capture any launch‑time errors (e.g., invalid configuration).
2. Call `cudaDeviceSynchronize()` to wait for the kernel to finish and catch any runtime errors that occur during execution.

If either call reports an error, the helper should print a descriptive message and terminate the program, so that the user immediately knows something went wrong.

The code also needs to illustrate the usage of this helper. A minimal “dummy” kernel is provided, launched in `main()`, followed by a call to `checkKernelLaunch()`. The program includes the necessary headers, uses `std::cerr` for error messages, and returns `EXIT_FAILURE` when an error is detected. This setup demonstrates how a developer can wrap every kernel launch with `checkKernelLaunch()` to catch both launch‑time and execution‑time errors in a clean, reusable way.

Note: The code is intentionally kept simple for clarity and portability. It compiles with any recent CUDA toolkit and can be extended with more complex kernels as needed. 
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

/**
 * @brief Helper function to check kernel launch and execution errors.
 *
 * This function should be called immediately after a kernel launch to ensure
 * that the kernel was launched successfully and executed without asynchronous
 * errors. It performs the following steps:
 *
 * 1. Checks for launch-time errors using cudaGetLastError().
 * 2. Waits for the kernel to finish and checks for execution-time errors
 *    using cudaDeviceSynchronize().
 *
 * If any error is detected, it prints an error message to std::cerr and exits
 * the program with EXIT_FAILURE.
 */
void checkKernelLaunch()
{
    // Check for launch-time errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Wait for device to finish and check for execution-time errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief A simple dummy kernel used for demonstration purposes.
 *
 * The kernel does nothing but demonstrates a valid kernel launch.
 */
__global__ void dummyKernel()
{
    // No operation; just a placeholder
}

int main()
{
    // Launch the dummy kernel with 1 block of 1 thread
    dummyKernel<<<1, 1>>>();

    // Immediately check for errors
    checkKernelLaunch();

    std::cout << "Kernel executed successfully." << std::endl;
    return EXIT_SUCCESS;
}
