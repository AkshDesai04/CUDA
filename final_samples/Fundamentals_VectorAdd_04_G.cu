```cpp
/*
Implement for character arrays: `C[i] = A[i] - B[i]`.

Thought process:
- The goal is to perform an element‑wise subtraction of two character arrays A and B
  and store the result in array C.  In C/C++ subtracting two `char` values yields an
  `int`, but the problem statement specifies that the result should be stored in a
  character array, so we cast the difference back to `char`.  This is exactly what
  the kernel below does.

- A CUDA kernel is required because the instruction explicitly asks for a CUDA
  program.  The kernel `subtract_chars` receives pointers to device memory for
  A, B, and C along with the array length.  Each thread processes one element,
  performs `C[i] = A[i] - B[i]`, and writes the result back.

- For a simple demonstration, the host code creates three small character arrays
  of length 8, initializes A and B with example data, allocates device memory,
  copies the host arrays to the device, launches the kernel, copies the result
  back, and prints the original and resulting arrays in hexadecimal form so
  that subtraction results can be verified.

- Error checking is performed after each CUDA API call to help catch
  failures early.

- The code compiles with `nvcc` and runs on any GPU capable of supporting the
  requested launch configuration.

- The program is self‑contained; no external files are required.

- The final output of the program will display the input arrays and the
  resulting array after subtraction.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>

// CUDA kernel for element-wise subtraction of two char arrays
__global__ void subtract_chars(const char *A, const char *B, char *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Perform subtraction and cast back to char
        C[idx] = static_cast<char>(A[idx] - B[idx]);
    }
}

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",        \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),             \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void)
{
    const size_t N = 8;                      // Array length
    char h_A[N] = { 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48 }; // 'A'..'H'
    char h_B[N] = { 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37 }; // '0'..'7'
    char h_C[N] = {0};

    // Device pointers
    char *d_A = nullptr;
    char *d_B = nullptr;
    char *d_C = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_A, N * sizeof(char)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, N * sizeof(char)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, N * sizeof(char)));

    // Copy host arrays to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(char), cudaMemcpyHostToDevice));

    // Launch kernel
    const size_t threadsPerBlock = 256;
    const size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    subtract_chars<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for kernel to finish and check for errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(char), cudaMemcpyDeviceToHost));

    // Print results
    printf("A: ");
    for (size_t i = 0; i < N; ++i) printf("%02X ", (unsigned char)h_A[i]);
    printf("\nB: ");
    for (size_t i = 0; i < N; ++i) printf("%02X ", (unsigned char)h_B[i]);
    printf("\nC (A - B): ");
    for (size_t i = 0; i < N; ++i) printf("%02X ", (unsigned char)h_C[i]);
    printf("\n");

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```