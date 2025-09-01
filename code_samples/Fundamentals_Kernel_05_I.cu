/*
Calculate the global ID starting from 1 instead of 0.
The program demonstrates how to compute a 1‑based global ID for each thread in a
CUDA kernel.  The usual CUDA idiom for a global thread index is:
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
which yields a 0‑based index.  To shift this to start from 1 we simply add
1 to the computed index before storing or using it.

My thought process:
1. Define a simple kernel that receives a pointer to an integer array and the
   number of elements N.
2. Inside the kernel compute the 0‑based global ID as described above.
3. Check that the ID is within bounds (gid < N) before accessing the array.
4. Store gid + 1 into the array, thereby giving each element a 1‑based ID.
5. In the host code allocate an array of size N, copy it to device memory,
   launch the kernel with enough blocks/threads to cover all N elements,
   copy the result back, and print the IDs to verify the 1‑based numbering.
6. Clean up allocated memory and return.

This straightforward example shows how to adjust the typical
0‑based indexing to 1‑based indexing in CUDA.

*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void computeGlobalId(int *output, int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x; // 0‑based global ID
    if (gid < N)
    {
        output[gid] = gid + 1; // shift to 1‑based
    }
}

int main(void)
{
    const int N = 16; // number of elements / threads
    int *h_output = (int *)malloc(N * sizeof(int));
    int *d_output;

    // Allocate device memory
    cudaMalloc((void **)&d_output, N * sizeof(int));

    // Determine block and grid dimensions
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    computeGlobalId<<<blocks, threadsPerBlock>>>(d_output, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        free(h_output);
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("1‑based global IDs:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("Thread %d -> ID %d\n", i, h_output[i]);
    }

    // Clean up
    cudaFree(d_output);
    free(h_output);

    return 0;
}
