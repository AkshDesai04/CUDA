/*
Aim: In a program with multiple, sequential kernel launches, place the check after each one to pinpoint which one is failing.

Thinking:
- The goal of this example is to demonstrate how to perform error checking after each kernel launch in a CUDA program that contains multiple sequential kernels.
- Each kernel will perform a simple vector operation on an array of floats.
- After launching a kernel, we immediately call cudaGetLastError() to capture any launch errors, and then cudaDeviceSynchronize() to wait for completion and capture any execution errors.
- If an error occurs, we print the kernel name, the error code, and a message, then exit the program.
- This approach allows us to identify exactly which kernel caused a failure.
- We also include a helper macro for readability.
- Memory is allocated on both host and device, and freed before program exit.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void addKernel(float *d_out, const float *d_in, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] + val;
    }
}

__global__ void mulKernel(float *d_out, const float *d_in, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] * val;
    }
}

__global__ void subKernel(float *d_out, const float *d_in, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] - val;
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Host memory
    float *h_in  = (float *)malloc(size);
    float *h_out = (float *)malloc(size);

    // Initialize input
    for (int i = 0; i < N; ++i) h_in[i] = (float)i;

    // Device memory
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void **)&d_in,  size));
    CHECK_CUDA(cudaMalloc((void **)&d_out, size));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 1. First kernel: add 1.0f
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N, 1.0f);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "addKernel launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "addKernel execution error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // 2. Second kernel: multiply by 2.0f
    mulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_out, N, 2.0f);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "mulKernel launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "mulKernel execution error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // 3. Third kernel: subtract 3.0f
    subKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_out, N, 3.0f);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "subKernel launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "subKernel execution error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    for (int i = 0; i < 10; ++i) {
        float expected = ((float)i + 1.0f) * 2.0f - 3.0f;
        if (h_out[i] != expected) {
            fprintf(stderr, "Mismatch at index %d: expected %f, got %f\n",
                    i, expected, h_out[i]);
            return EXIT_FAILURE;
        }
    }
    printf("All tests passed.\n");

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return EXIT_SUCCESS;
}
