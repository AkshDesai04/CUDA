/*
Make the macro a `do { ... } while(0)` block to ensure it behaves like a single statement.

Thought process:
- The user explicitly requested that a macro be defined as a `do { ... } while(0)` block.  
- In CUDA, macros are typically used both on the host side and within device kernels.  
- To demonstrate a simple, useful macro that behaves like a single statement, I chose a macro that
  performs an addition and stores the result in a third variable: `ADD_RESULT(a,b,c)`.  
- Using `do { ... } while(0)` ensures that the macro expands into a block that can safely be
  used in an `if` statement without needing braces.  
- I wrapped this macro in a small CUDA program that allocates two vectors, copies them to the
  device, runs a kernel that adds the vectors element‑wise using the macro, copies the result
  back, and prints it.  
- The code includes basic error handling by synchronizing the device after the kernel launch
  and freeing device memory before exiting.  
- The program is self‑contained, starts from the `#include` directives, defines the macro,
  declares the kernel, implements `main`, and ends with a `return 0;`.  
- No additional files or external dependencies are required; the program compiles as a
  single `.cu` file with a CUDA compiler such as `nvcc`.  
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Macro that performs addition and stores the result in a third variable.
   The do { ... } while(0) wrapper ensures it behaves like a single statement. */
#define ADD_RESULT(a, b, c) do { (c) = (a) + (b); } while(0)

/* Device kernel that adds two input arrays element‑wise and writes the result to the output array. */
__global__ void vectorAdd(const int *A, const int *B, int *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        ADD_RESULT(A[idx], B[idx], C[idx]);
    }
}

int main()
{
    const int N = 10;
    size_t size = N * sizeof(int);

    /* Host arrays */
    int h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = i;
        h_B[i] = N - i;
    }

    /* Device arrays */
    int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    /* Copy inputs to device */
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    /* Copy result back to host */
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    /* Print results */
    printf("Result:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    /* Clean up */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
