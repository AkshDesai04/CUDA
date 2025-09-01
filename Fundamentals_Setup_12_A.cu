/*
Create a syntax error by misspelling __global__ as __globol__ and observe the compiler error.

Thought process: The task is to deliberately introduce a syntax error in a CUDA program by misspelling the CUDA kernel qualifier __global__. The goal is to produce a compilation error that highlights the incorrect keyword usage. To achieve this, I will:
1. Add a multiline comment at the top that contains the exact phrase from the prompt and my internal reasoning.
2. Include necessary headers (<stdio.h>).
3. Define a kernel function with the misspelled qualifier __globol__.
4. Allocate a simple integer array on the device, call the kernel, and copy the result back to the host.
5. Print the result to verify normal execution (though the kernel launch will fail to compile).
6. Keep the code self-contained and valid except for the intentional typo, so the compiler error is the only issue.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Dummy kernel with misspelled __global__ qualifier
__globol__ void addOne(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] += 1;
    }
}

int main(void) {
    const int N = 10;
    size_t size = N * sizeof(int);
    int h_arr[N];

    // Initialize host array
    for (int i = 0; i < N; ++i) h_arr[i] = i;

    // Allocate device memory
    int *d_arr;
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // Launch kernel (will not compile due to misspelled qualifier)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addOne<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Copy back result
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Result array: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    return 0;
}
