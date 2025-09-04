/*
Use Nsight Systems (`nsys`) to visualize the execution timeline.
You should see a serial pattern: [COPY]...[KERNEL]...[COPY]...

This program demonstrates a typical CUDA workflow where we allocate memory on host, transfer data to device (COPY), launch a kernel (KERNEL), and transfer results back (COPY). By running Nsight Systems with this code, one should observe the serial execution of copy, kernel, and copy events in the timeline, confirming that no overlapping of these stages occurs due to default synchronization behavior.

The program uses a simple vector addition kernel to illustrate the copy and kernel phases. It performs allocation, initialization, memory transfer, kernel launch, memory copy back, and cleanup. No streams or asynchronous operations are used, ensuring the timeline shows a clear sequential pattern.

The code includes basic error checking via cudaGetLastError after kernel launch and cudaGetDeviceProperties for demonstration. It also uses the cudaDeviceSynchronize to wait for the kernel to finish before copying back, reinforcing the serial order.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Host memory allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device (COPY)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel (KERNEL)
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Synchronize device to ensure kernel completion
    cudaDeviceSynchronize();

    // Copy result back to host (COPY)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Simple verification
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            printf("Mismatch at index %d: host=%f, device=%f\n", i, h_A[i] + h_B[i], h_C[i]);
            break;
        }
    }
    if (correct) {
        printf("Vector addition successful.\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
