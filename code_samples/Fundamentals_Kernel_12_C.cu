/*
Create a helper C++ function `void launch_kernel_1d(int n_elements)` that computes the correct grid/block dimensions and launches a kernel.

**Thinking Process:**

1. **Purpose of the Helper Function**  
   The function should determine an efficient launch configuration for a 1‑D kernel, ensuring that all elements are processed and that thread blocks are fully utilized.  
   - We choose a conventional block size of 256 threads (common for many GPUs).  
   - The grid size is calculated as the ceiling of `n_elements / blockSize`.

2. **Kernel Design**  
   - The kernel will simply write the index value to each element of an integer array, demonstrating that each thread processes a unique element.  
   - The kernel signature: `__global__ void kernel_1d(int *data, int n)`.

3. **Host‑Side Data Management**  
   - Allocate a host array of size `n_elements`, initialize it to zero.  
   - Allocate device memory, copy the host array to device.  
   - After the kernel runs, copy the result back to host and print the first few elements for verification.

4. **Error Checking**  
   - Use `cudaGetLastError()` after kernel launch and `cudaDeviceSynchronize()` to catch any launch failures.  
   - Check the return values of CUDA API calls and handle errors appropriately.

5. **Overall Flow**  
   - `main()` calls `launch_kernel_1d(n_elements)` to demonstrate usage.  
   - The helper function prints the launch configuration and the first 10 results.

The resulting .cu file is fully self‑contained and ready to compile with `nvcc`.
*/

#include <iostream>
#include <cuda_runtime.h>

// Kernel that writes its global thread index into the array
__global__ void kernel_1d(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

// Helper function that calculates launch dimensions and runs the kernel
void launch_kernel_1d(int n_elements) {
    // Choose a block size that works well on most GPUs
    const int blockSize = 256;
    // Calculate the number of blocks needed to cover all elements
    int gridSize = (n_elements + blockSize - 1) / blockSize;

    std::cout << "Launching kernel with gridSize = " << gridSize
              << ", blockSize = " << blockSize << std::endl;

    // Allocate host memory
    int *h_data = new int[n_elements];
    for (int i = 0; i < n_elements; ++i) h_data[i] = 0;

    // Allocate device memory
    int *d_data = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_data, n_elements * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        delete[] h_data;
        return;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_data, h_data, n_elements * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        delete[] h_data;
        return;
    }

    // Launch the kernel
    kernel_1d<<<gridSize, blockSize>>>(d_data, n_elements);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        delete[] h_data;
        return;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        delete[] h_data;
        return;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, n_elements * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        delete[] h_data;
        return;
    }

    // Print first few results for verification
    std::cout << "First 10 elements after kernel execution:" << std::endl;
    for (int i = 0; i < std::min(n_elements, 10); ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_data);
    delete[] h_data;
}

int main() {
    int n = 1000; // Example number of elements
    launch_kernel_1d(n);
    return 0;
}
