```cpp
/* 
Aim of the program: 
Write the following program for CUDA with c and the output file should be a .cu file. 

Thinking:
The user’s request is to provide a complete CUDA C program in a single .cu file, 
including a multiline comment at the top that repeats the aim verbatim and also 
contains my entire reasoning.  
The program will contain a simple kernel that performs an element‑wise addition of
two arrays, a host function that sets up device memory, launches the kernel,
and then copies the result back to the host.  Basic error checking is included.
This will satisfy the developer instruction to output a valid .cu file with
no extraneous text or supporting files. 
*/

#include <cuda_runtime.h>
#include <iostream>

// Simple kernel that adds two arrays elementwise
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for launch errors
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result for a few elements
    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": "
                      << h_C[i] << " != " << expected << std::endl;
            break;
        }
    }
    if (correct) {
        std::cout << "Vector addition successful." << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
```