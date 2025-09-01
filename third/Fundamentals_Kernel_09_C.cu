/*
Aim: Explain the difference between __global__ and __device__ functions in a comment.

Thought process:
1. The user requested an explanation of the differences between __global__ and __device__ functions.  
2. According to the developer instruction, I must produce a .cu file that starts with a multiline comment containing the aim verbatim, followed by my entire thinking.  
3. The comment will therefore include the user’s sentence and my reasoning about how to demonstrate the difference.  
4. To provide a concrete illustration, I will write a minimal CUDA program that defines:
   - a __device__ function `add` that performs integer addition, callable only from device code.
   - a __global__ kernel `vectorAdd` that calls `add` and writes the result to device memory.
   - a host `main` that allocates vectors, copies data to the device, launches the kernel, copies the result back, and prints it.
5. The program will compile with `nvcc` and run on any CUDA-capable GPU, demonstrating the calling conventions and visibility rules of the two function qualifiers.
6. I will include sufficient comments in the code to explain each part, but keep the file concise and self-contained.

Now the complete CUDA source file follows.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* 
 * __device__ function: Can only be called from device (kernel) code.
 * It runs on the GPU and cannot be invoked from host code.
 * It has limited visibility; it cannot launch kernels or access host memory.
 */
__device__ int add(int a, int b) {
    return a + b;
}

/* 
 * __global__ kernel: Callable from host. It runs on the GPU.
 * The launch syntax uses <<<gridDim, blockDim>>>. 
 * It can call __device__ functions, launch other kernels (in newer CUDA), and read/write global memory.
 */
__global__ void vectorAdd(const int *A, const int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = add(A[idx], B[idx]);  // Call to __device__ function
    }
}

int main(void) {
    const int N = 5;
    size_t size = N * sizeof(int);

    int h_A[N] = {1, 2, 3, 4, 5};
    int h_B[N] = {10, 20, 30, 40, 50};
    int h_C[N] = {0};

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result of vector addition:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
