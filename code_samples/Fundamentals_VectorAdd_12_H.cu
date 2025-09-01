```c
/*
 * Aim: Verify the matrix addition result on the CPU.
 *
 * Thinking:
 * 1. I will write a CUDA program that performs matrix addition on the GPU.
 * 2. The program will allocate two input matrices A and B and one output matrix C on the host.
 * 3. I will copy A and B to the device, launch a simple element‑wise addition kernel,
 *    then copy the result matrix C back to the host.
 * 4. On the CPU I will compute the reference result by adding A and B element‑wise.
 * 5. I will compare each element of the GPU result with the CPU reference.
 * 6. If all elements match, the program prints "Verification passed."
 *    Otherwise, it reports the first mismatch.
 * 7. I will include simple CUDA error checking macros and make the code
 *    self‑contained, compiling with `nvcc` to produce a `.cu` file.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__       \
                      << " : " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void matrixAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1024;                     // Matrix dimension N x N
    const int size = N * N;
    const int bytes = size * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(size), h_B(size), h_C(size), h_C_ref(size);

    // Initialize host matrices A and B with some values
    for (int i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(i % 100);
        h_B[i] = static_cast<float>((i * 2) % 100);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    for (int i = 0; i < size; ++i) {
        h_C_ref[i] = h_A[i] + h_B[i];
    }

    // Verify results
    bool match = true;
    for (int i = 0; i < size; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": GPU result "
                      << h_C[i] << " != CPU result " << h_C_ref[i] << std::endl;
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Verification passed." << std::endl;
    } else {
        std::cout << "Verification failed." << std::endl;
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```