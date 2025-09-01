/*
Aim:
Write a comment in your code showing the full `nvcc` command used to compile it.

Thinking:
The user’s requirement is to provide a CUDA program that contains a comment with the exact nvcc command used to compile it. 
To satisfy this, I’ll write a complete, self‑contained CUDA C program that performs a simple vector addition on the GPU. 
The program will:

1. Allocate two input vectors and one output vector on the host.
2. Allocate corresponding memory on the device.
3. Copy the input vectors to the device.
4. Launch a kernel that adds the vectors element‑wise.
5. Copy the result back to the host.
6. Verify a few elements for correctness and free all allocated memory.

The code will include:
- Standard CUDA headers and error checking macro.
- A simple kernel `vecAdd` that sums two float arrays.
- Host code that sets up data, launches the kernel, and prints a short verification.

Additionally, as requested, I’ll place a single‑line comment that shows the exact nvcc command needed to compile this file. This comment will be placed after the code for clarity, but it could equally be inside the initial multiline comment. The command will target a common compute capability (e.g., sm_61) and enable optimizations with `-O2`. No extraneous text will be output—only the .cu source code.

*/

// CUDA kernel for vector addition
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Macro for checking CUDA errors following a kernel launch or API call
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",        \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Host allocations
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device allocations
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 5; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: host=%f device=%f\n", i, expected, h_C[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful for first 5 elements.\n");
    }

    // Free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

// nvcc -o vector_add vector_add.cu -arch=sm_61 -O2