/*
Why is it good practice to check for errors both immediately after launch and at synchronization? (Catch config errors early, catch execution errors later).

The aim of this program is to demonstrate why checking for CUDA errors right after a kernel launch and again after synchronization is important. 
CUDA kernels are launched asynchronously, meaning the launch call returns before the kernel has actually executed. 
If the launch configuration is incorrect (e.g., launching more blocks/threads than supported, passing invalid pointers, or setting an illegal grid/block dimension), the error will not surface until a later API call that forces the GPU to finish pending work. 
Checking immediately after launch with cudaGetLastError() lets us catch configuration or launch‑time errors early, before any subsequent operations that depend on the kernel’s result. 
Conversely, runtime errors that occur during the kernel’s execution (e.g., illegal memory accesses, division by zero, or other exceptions) will not be reported until the GPU completes the work. 
Synchronizing with cudaDeviceSynchronize() forces the host to wait for the kernel to finish and then returns any errors that occurred during execution. 
Thus, checking both after launch and after synchronization provides comprehensive coverage: early detection of launch configuration problems and late detection of execution problems. 
This two‑step error checking pattern is a good practice for robust CUDA code.
*/

#include <cuda_runtime.h>
#include <iostream>

// Simple kernel that increments each element of an array
__global__ void incrementKernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Intentional out‑of‑bounds access for demonstration when n is too small
        d_arr[idx] += 1;
    }
}

// Helper macro to check CUDA errors after a call
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error (" << err << ") at "           \
                      << __FILE__ << ":" << __LINE__ << " - "       \
                      << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main()
{
    const int N = 10;
    const int SIZE = N * sizeof(int);

    // Allocate host memory
    int *h_arr = new int[N];
    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    // Allocate device memory
    int *d_arr = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, SIZE));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, SIZE, cudaMemcpyHostToDevice));

    // Define launch configuration
    const int threadsPerBlock = 4;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for launch errors (config errors)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_arr);
        delete[] h_arr;
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Kernel launch succeeded (no config errors)." << std::endl;
    }

    // Synchronize to catch execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel execution error after sync: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_arr);
        delete[] h_arr;
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Kernel execution succeeded (no runtime errors)." << std::endl;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, SIZE, cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Result array:" << std::endl;
    for (int i = 0; i < N; ++i)
        std::cout << h_arr[i] << " ";
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_arr));
    delete[] h_arr;

    return EXIT_SUCCESS;
}
