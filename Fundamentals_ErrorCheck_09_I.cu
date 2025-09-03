/*
Aim: Convert your `checkKernelLaunch` helper function to throw instead of exit.

Thought process:
- The original helper function likely used `cudaGetLastError()` to query the status of the most recent kernel launch.  
- On encountering an error, it printed a message and called `exit(EXIT_FAILURE)`.  
- Replacing the exit with an exception allows the caller to decide how to handle the error (e.g., cleanup, retry, propagate).  
- In C++ we can throw a `std::runtime_error` containing the kernel name and the CUDA error string.  
- We need to include `<stdexcept>` and `<string>` to construct the exception.  
- The rest of the program remains largely unchanged: launch a kernel, call `checkKernelLaunch`, and catch the exception in `main`.  
- The example demonstrates a simple addition kernel on two small arrays, providing context for the helper function usage.  
- All CUDA API calls should be checked, but for brevity only the kernel launch is wrapped; error handling for memory allocation/transfer could be added similarly.  
- The program is self‑contained and compiles as a single `.cu` file.  
*/

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

/* Helper function that checks the result of the last kernel launch.
   Throws a std::runtime_error instead of exiting the program. */
void checkKernelLaunch(const char* name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(name) + " failed: " + cudaGetErrorString(err));
    }
}

/* Simple kernel that adds two arrays element‑wise. */
__global__ void addKernel(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 10;
    int h_a[N], h_b[N], h_c[N];

    /* Initialize host data */
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    /* Allocate device memory */
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    /* Copy data to device */
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    /* Check for kernel launch errors */
    try {
        checkKernelLaunch("addKernel");
    } catch (const std::exception& e) {
        std::cerr << "Kernel error: " << e.what() << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return EXIT_FAILURE;
    }

    /* Copy result back to host */
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    /* Print results */
    std::cout << "Result: ";
    for (int i = 0; i < N; ++i) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    /* Clean up */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
