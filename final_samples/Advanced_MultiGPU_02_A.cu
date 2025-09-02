/*
After the kernels complete, copy the results from both GPUs back to the host and assemble the final complete vector C.

Thought process:
1. The aim of this program is to demonstrate multi‑GPU usage in CUDA by performing a simple element‑wise addition of two large vectors.
2. We will split the input vectors into two equal halves, each processed by a separate GPU.
3. For each GPU we will:
   - Set the device using cudaSetDevice.
   - Allocate device memory for its portion of vectors A, B and the result C.
   - Copy the corresponding sub‑arrays from host to device.
   - Launch a kernel that adds the two sub‑arrays.
   - Copy the partial result back to host memory.
4. After both GPUs finish, the host will assemble the two partial results into the final vector C.
5. Error checking will be done after each CUDA call to ensure correctness.
6. The kernel is simple: C[i] = A[i] + B[i].
7. We use CUDA streams for asynchronous copies but keep it straightforward; the main flow is sequential to simplify reasoning.
8. The program requires at least two GPUs; otherwise it exits with an error.
9. We will test the program with a default vector size of 1<<20 (about one million elements) but this can be changed via command line.
10. The final output will print the first 10 elements of the result vector to verify correctness.

The code below follows this design and is fully self‑contained in a single .cu file.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main(int argc, char **argv)
{
    // Determine number of GPUs
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2)
    {
        fprintf(stderr, "Error: This program requires at least 2 GPUs.\n");
        return EXIT_FAILURE;
    }

    // Vector size (default 1<<20)
    size_t N = 1 << 20;
    if (argc > 1)
    {
        N = atoll(argv[1]); // parse size from command line
    }

    // Allocate host memory
    size_t bytes = N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input vectors
    for (size_t i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Split data into two halves
    size_t N_half = N / 2;
    size_t bytes_half = N_half * sizeof(float);

    // Device 0
    CUDA_CHECK(cudaSetDevice(0));
    float *d_A0, *d_B0, *d_C0;
    CUDA_CHECK(cudaMalloc((void**)&d_A0, bytes_half));
    CUDA_CHECK(cudaMalloc((void**)&d_B0, bytes_half));
    CUDA_CHECK(cudaMalloc((void**)&d_C0, bytes_half));
    CUDA_CHECK(cudaMemcpy(d_A0, h_A, bytes_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B0, h_B, bytes_half, cudaMemcpyHostToDevice));

    // Device 1
    CUDA_CHECK(cudaSetDevice(1));
    float *d_A1, *d_B1, *d_C1;
    CUDA_CHECK(cudaMalloc((void**)&d_A1, bytes_half));
    CUDA_CHECK(cudaMalloc((void**)&d_B1, bytes_half));
    CUDA_CHECK(cudaMalloc((void**)&d_C1, bytes_half));
    CUDA_CHECK(cudaMemcpy(d_A1, h_A + N_half, bytes_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, h_B + N_half, bytes_half, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N_half + threadsPerBlock - 1) / threadsPerBlock;

    // Launch on device 0
    CUDA_CHECK(cudaSetDevice(0));
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A0, d_B0, d_C0, N_half);
    CUDA_CHECK(cudaGetLastError());

    // Launch on device 1
    CUDA_CHECK(cudaSetDevice(1));
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A1, d_B1, d_C1, N_half);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(h_C, d_C0, bytes_half, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(h_C + N_half, d_C1, bytes_half, cudaMemcpyDeviceToHost));

    // Assemble final vector C (already assembled in h_C)

    // Verify first 10 elements
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Cleanup
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d_A0));
    CUDA_CHECK(cudaFree(d_B0));
    CUDA_CHECK(cudaFree(d_C0));

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_C1));

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Computation completed successfully.\n");
    return EXIT_SUCCESS;
}
