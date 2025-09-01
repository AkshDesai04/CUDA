/*
Set up a 2D problem for a 512x512 image. Define a 2D block size of 16x16. Calculate the 2D grid dimensions needed. Create dim3 variables for both grid and block and launch the kernel.

Thinking:
- We need a complete CUDA .cu file that demonstrates setting up a 2D problem.
- The image size is 512x512, so we will allocate an array of that size.
- The block size is 16x16, so each block will contain 256 threads.
- Grid dimensions: ceil(512/16) = 32 in each dimension.
- We will create dim3 block(16, 16) and dim3 grid(32, 32).
- Launch a simple kernel that writes the linear index to each pixel or does some trivial computation.
- Include basic memory allocation and deallocation, error checking, and a simple host print to confirm execution.
- Keep the code minimal but compilable and self‑contained.
*/

#include <cuda_runtime.h>
#include <iostream>

// Simple kernel that writes the global thread index into the output array.
__global__ void processImageKernel(int* d_output, int width, int height)
{
    // Compute the 2D index of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    // Bounds check
    if (x < width && y < height)
    {
        int idx = y * width + x;
        // For demonstration, write the linear index
        d_output[idx] = idx;
    }
}

int main()
{
    const int width  = 512;
    const int height = 512;
    const size_t numPixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t sizeInBytes = numPixels * sizeof(int);

    // Allocate host memory
    int* h_output = new int[numPixels];

    // Allocate device memory
    int* d_output = nullptr;
    cudaError_t err = cudaMalloc(&d_output, sizeInBytes);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        delete[] h_output;
        return EXIT_FAILURE;
    }

    // Define block and grid dimensions
    dim3 block(16, 16);                     // 16x16 threads per block
    dim3 grid((width + block.x - 1) / block.x, // ceil division
              (height + block.y - 1) / block.y);

    // Launch the kernel
    processImageKernel<<<grid, block>>>(d_output, width, height);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output);
        delete[] h_output;
        return EXIT_FAILURE;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output);
        delete[] h_output;
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_output, d_output, sizeInBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output);
        delete[] h_output;
        return EXIT_FAILURE;
    }

    // Simple verification: print first 10 pixel values
    std::cout << "First 10 pixel values:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_output);
    delete[] h_output;

    return EXIT_SUCCESS;
}
