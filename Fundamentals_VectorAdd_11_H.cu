/*
Modify the grid-stride loop to process elements backwards from `N-1` down to 0.

The task is to write a CUDA program that demonstrates a grid‑stride loop that processes an array
in reverse order, i.e. starting from the last element (index N‑1) and moving towards the first
element (index 0).  The core idea is to calculate the starting global index for each thread
in reverse, and then decrement the index by the full grid stride (blockDim.x * gridDim.x)
in each iteration of the loop.

Key points considered while designing this program:

1. **Index calculation** – For a forward grid‑stride loop we usually use  
   `int idx = blockIdx.x * blockDim.x + threadIdx.x;` and then iterate with  
   `idx += stride`.  To traverse backwards we instead compute  
   `int idx = N-1 - (blockIdx.x * blockDim.x + threadIdx.x);`.  
   This gives each thread a starting point that is a certain offset from the end.

2. **Loop termination** – We need a condition that stops the loop when `idx` becomes
   negative.  Using a signed integer (`int`) for `idx` allows us to simply test  
   `idx >= 0`.  If an unsigned type were used the comparison would wrap around.

3. **Stride** – The stride remains the same as in the forward case:  
   `int stride = blockDim.x * gridDim.x;`.  In each loop iteration we do  
   `idx -= stride`.

4. **Kernel logic** – For demonstration we will simply write the value of the index
   into the array.  The effect is that after the kernel finishes, `data[i]` will
   contain `i` for all `i` in `0..N-1`.  The reverse traversal does not change the
   final result because the kernel writes to the exact same index it reads, but it
   illustrates how the loop iterates in descending order.

5. **Host code** – The host program allocates an integer array on the device,
   initializes it to zero, launches the kernel, copies the result back to the host,
   and prints a few elements from the beginning and the end to verify correctness.

6. **Error checking** – A simple macro `CUDA_CHECK` is used to abort if any CUDA
   API call fails.

7. **Compile** – The code is intended to be compiled with `nvcc` as a `.cu` file.

Overall, this program is a minimal, self‑contained example that shows how to
implement a reverse grid‑stride loop in CUDA.
*/

#include <stdio.h>
#include <stdlib.h>

#define N 1024              // Size of the array
#define BLOCK_SIZE 256      // Threads per block

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel that writes the index value into the array in reverse order
__global__ void reverseKernel(int *data, int n) {
    int stride = blockDim.x * gridDim.x;
    // Compute the starting index for this thread in reverse
    int idx = n - 1 - (blockIdx.x * blockDim.x + threadIdx.x);

    for (int i = idx; i >= 0; i -= stride) {
        data[i] = i;  // For demonstration: store the index itself
    }
}

int main(void) {
    int *d_data = NULL;
    int *h_data = NULL;

    size_t size = N * sizeof(int);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_data, size));

    // Allocate host memory
    h_data = (int *)malloc(size);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array to zeros
    for (int i = 0; i < N; ++i) {
        h_data[i] = 0;
    }

    // Copy initial data to device (optional, here just to show transfer)
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel with enough blocks to cover all elements
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reverseKernel<<<numBlocks, BLOCK_SIZE>>>(d_data, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Print a few elements to verify
    printf("First 10 elements:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    printf("Last 10 elements:\n");
    for (int i = N - 10; i < N; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return 0;
}