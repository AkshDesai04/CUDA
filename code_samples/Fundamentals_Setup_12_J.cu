/*
Aim: Compile with the `-v` (verbose) flag to see the internal steps `nvcc` takes, including where it separates host and device code.

Thinking:
1. The user wants a CUDA program that can be compiled with `-v` to observe `nvcc`'s internal steps.  
2. A minimal program that includes both host and device code is a simple vector addition kernel.  
3. The program will:
   - Allocate device memory for two input arrays and one output array.
   - Copy input data from host to device.
   - Launch a kernel that adds the two arrays element‑wise.
   - Copy the result back to host and print it.
4. The code is written in plain CUDA C, uses only the runtime API, and contains basic error checking.  
5. When compiled with `nvcc -v file.cu`, the output will show the preprocessing, compilation, and linking stages, and it will indicate where the host and device code are separated (e.g., separate PTX/IR files).  
6. No additional libraries or external files are required; the program is self‑contained and can be built on any system with a CUDA toolkit installed.  
7. The comment block at the top contains the exact aim statement as requested, followed by a detailed explanation of the design choices and expected compiler behavior.  
8. The rest of the file is the actual CUDA program.  
9. Users can compile with `nvcc -v example.cu -o example` and observe the verbose output.  
10. The program uses a fixed array size for simplicity; this keeps the code short and easy to understand.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10
#define CHECK_CUDA(call)                                       \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",  \
                    #call, __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));                   \
            return EXIT_FAILURE;                                \
        }                                                       \
    } while (0)

__global__ void vectorAdd(const int *A, const int *B, int *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    int h_A[N], h_B[N], h_C[N];
    int *d_A = NULL, *d_B = NULL, *d_C = NULL;
    size_t size = N * sizeof(int);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    /* Copy inputs to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Print results */
    printf("A\tB\tC\n");
    for (int i = 0; i < N; ++i) {
        printf("%d\t%d\t%d\n", h_A[i], h_B[i], h_C[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return EXIT_SUCCESS;
}
