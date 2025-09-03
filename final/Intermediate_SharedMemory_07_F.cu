```cuda
/*
Aim: Write a kernel to shift (rotate) an array by `k` positions using shared memory.

Thinking:
1. The goal is to rotate an array of length `n` by `k` positions to the right.
   After rotation, element originally at index `i` should move to index
   `(i + k) % n`.  In terms of writing back to the original array, we
   need to assign each destination index `dst = i` the value that was
   originally at `src = (i + k) % n`.

2. Shared memory is used for the rotation to minimize global memory
   traffic.  The kernel loads the entire array into a shared memory
   buffer `s_arr`.  Each thread then writes the rotated value back to
   global memory.

3. The kernel is launched with a single block containing `n` threads.
   This works for arrays that fit into shared memory.  For larger
   arrays, the kernel could be extended to load segments or use a
   multi‑block strategy, but that would complicate the example.

4. Because shared memory is limited, we enforce that the array size
   `n` does not exceed the shared memory available for a block
   (commonly 48KB).  The example uses a small array of 10 elements
   to stay well within limits.

5. We handle the case where `k` may be larger than `n` by taking
   `k % n`.  Negative values of `k` are not supported in this
   simplified example.

6. Error checking macros (`CUDA_CHECK`) are used to catch API errors.

7. After the kernel completes, the host code copies the rotated array
   back to host memory and prints the result.

8. The program is self‑contained and can be compiled with `nvcc`:
      nvcc -o rotate rotate.cu
   and run with `./rotate`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA API calls
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),    \
                    cudaGetErrorString(err));                          \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel to rotate an array by k positions to the right using shared memory
__global__ void rotateKernel(int *d_arr, int n, int k)
{
    // Shared memory buffer for the whole array
    extern __shared__ int s_arr[];

    int tid = threadIdx.x;

    // Load data from global to shared memory
    if (tid < n) {
        s_arr[tid] = d_arr[tid];
    }
    __syncthreads();

    // Compute source index for rotation (right shift)
    // After rotation, element at dst = tid should get value from src
    // where src = (tid + k) % n
    int src = (tid + k) % n;

    // Write rotated value back to global memory
    if (tid < n) {
        d_arr[tid] = s_arr[src];
    }
}

int main(void)
{
    const int N = 10;            // Array size
    const int K = 3;             // Rotation steps (right shift)

    int h_arr[N];
    int h_rotated[N];

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i + 1; // 1, 2, ..., N
    }

    // Allocate device memory
    int *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    // Compute effective rotation steps
    int k = K % N;

    // Launch kernel: one block, N threads, shared memory size N * sizeof(int)
    dim3 blockDim(N);
    dim3 gridDim(1);
    size_t sharedMemSize = N * sizeof(int);

    rotateKernel<<<gridDim, blockDim, sharedMemSize>>>(d_arr, N, k);
    CUDA_CHECK(cudaGetLastError());

    // Copy rotated array back to host
    CUDA_CHECK(cudaMemcpy(h_rotated, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Original array:  ");
    for (int i = 0; i < N; ++i) printf("%d ", h_arr[i]);
    printf("\n");

    printf("Rotated array:   ");
    for (int i = 0; i < N; ++i) printf("%d ", h_rotated[i]);
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
```