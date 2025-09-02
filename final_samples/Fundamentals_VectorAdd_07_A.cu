/*
Implement a kernel that cubes every element: `output[i] = input[i] * input[i] * input[i]`.

The goal of this program is to provide a complete, self‑contained CUDA C program that demonstrates a simple element‑wise operation on an array: cubing each element.  The kernel is intentionally straightforward – it reads a value from an input array, multiplies it by itself twice, and writes the result to an output array.  The host code allocates memory on both the host and device, initializes a small test input array, copies it to the device, launches the kernel, copies the results back, and finally prints the input and output arrays to verify correctness.  Basic error checking is included after each CUDA runtime call to make debugging easier.  The program compiles with `nvcc` and runs on any CUDA‑enabled device.

Key design choices:
1. **Kernel launch configuration**: a single block with a thread per element is sufficient for the small test data, but the code is written generically so that it can scale to larger arrays by using `gridDim` and `blockDim`.
2. **Memory allocation**: `cudaMalloc` for device memory and `cudaMemcpy` for data transfer.
3. **Error handling**: a helper macro `CUDA_CHECK` is used to wrap CUDA API calls.
4. **Testing**: a simple `int` array of 10 elements is used; the expected output is the cube of each integer.
5. **Portability**: the code includes necessary includes and can be compiled on any system with the CUDA toolkit installed.

This structure keeps the example minimal yet complete, making it a useful reference for beginners learning about writing and launching CUDA kernels that perform basic arithmetic operations. */

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for checking CUDA runtime API errors */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Kernel that cubes each element of the input array */
__global__ void cubeKernel(const int *input, int *output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = input[idx];
        output[idx] = val * val * val;
    }
}

int main(void)
{
    const size_t N = 10;               // Number of elements
    const size_t size = N * sizeof(int);

    /* Allocate and initialize host input array */
    int h_input[N];
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = (int)(i + 1);     // 1, 2, ..., 10
    }
    int h_output[N] = {0};

    /* Device pointers */
    int *d_input = NULL;
    int *d_output = NULL;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&d_input, size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    /* Determine block and grid sizes */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    cubeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Input  : ");
    for (size_t i = 0; i < N; ++i) {
        printf("%d ", h_input[i]);
    }
    printf("\n");

    printf("Output : ");
    for (size_t i = 0; i < N; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
