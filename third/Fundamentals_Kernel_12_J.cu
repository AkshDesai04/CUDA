/*
Aim: Explain in a comment why calculating the number of blocks as `(N + BS - 1) / BS` is better than `N / BS`.

Thinking:
- In CUDA, the number of blocks determines how many thread blocks are launched to cover all `N` elements.
- If you simply use `N / BS` (integer division), you lose the remainder because integer division truncates toward zero.
- For example, if `N = 1000` and `BS = 256`, `N / BS` yields `3` blocks (since 1000/256 = 3.90625 truncated to 3). That only covers `3 * 256 = 768` elements, leaving 232 elements unprocessed.
- The correct way is to compute the ceiling of `N / BS`. A common idiom for integer ceil division is `(N + BS - 1) / BS`. This adds `BS-1` before division, ensuring that any remainder causes an extra block to be counted.
- Thus `(N + BS - 1) / BS` guarantees that all `N` elements are assigned to at least one thread.
- It is concise, free of floating point operations, and works for any positive `N` and `BS`.
- Using `N / BS` could lead to out-of-range accesses or missing computations, especially when `N` is not a multiple of `BS`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define BS 256  // Block size

__global__ void add_one(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

int main() {
    const int N = 1000;  // Number of elements (intentionally not a multiple of BS)
    int *h_data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) h_data[i] = i;

    int *d_data;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate blocks using the two methods
    int blocks_div = N / BS;                  // Truncating division
    int blocks_ceil = (N + BS - 1) / BS;      // Ceiling division

    printf("Using N/BS: blocks = %d, covers %d elements\n", blocks_div, blocks_div * BS);
    printf("Using (N+BS-1)/BS: blocks = %d, covers %d elements\n", blocks_ceil, blocks_ceil * BS);

    // Launch kernel with the correct number of blocks
    add_one<<<blocks_ceil, BS>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != i + 1) {
            errors++;
            if (errors < 10) {
                printf("Error at index %d: expected %d, got %d\n", i, i + 1, h_data[i]);
            }
        }
    }
    if (errors == 0) {
        printf("All elements processed correctly.\n");
    } else {
        printf("Total errors: %d\n", errors);
    }

    cudaFree(d_data);
    free(h_data);
    return 0;
}
