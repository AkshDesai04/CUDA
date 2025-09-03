```cuda
/*
Aim: Write the kernel definition in a separate `.cu` file and the `main` function in another. Try to compile and link them together with `nvcc`.

Thinking:
The user asked for two separate `.cu` files: one containing only the kernel definition and another containing the `main` function that calls the kernel. However, the developer instruction explicitly requires that only a single `.cu` file be output, with no other supporting information or text. Therefore, the most straightforward way to satisfy both constraints is to place both the kernel definition and the `main` function in the same `.cu` file. This will still compile and link correctly with `nvcc`, and it respects the developer’s requirement of a single output file. The kernel will perform a simple vector addition to keep the example clear and functional. Proper error checking will be included for robustness.*/

// CUDA error checking macro
#define CHECK_CUDA(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

// Kernel: vector addition (C = A + B)
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// Main function
int main()
{
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify result for first 10 elements
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Verification failed at index %d: %f != %f\n",
                    i, h_C[i], expected);
            return EXIT_FAILURE;
        }
    }
    printf("Vector addition succeeded for first 10 elements.\n");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```