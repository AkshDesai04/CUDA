/*
Aim of the program: Modify the grid-stride loop to process elements backwards from N-1 down to 0.

Thinking:
To change a typical grid‑stride loop that iterates forward (i < N) to one that iterates backward, we must compute a starting index for each thread that is the highest index it should process. For a thread with global index `tid = blockIdx.x * blockDim.x + threadIdx.x`, the first element it handles is `N-1 - tid`. After that, it should step backwards by the total stride, `stride = blockDim.x * gridDim.x`. The loop condition therefore becomes `i >= 0`. We also guard against threads whose starting index would be negative (which happens when the total number of threads exceeds N). This ensures each element is processed exactly once, in reverse order, while preserving the grid‑stride parallelism.

Implementation details:
- The kernel `processBackward` multiplies each array element by 2 as a simple operation.
- Host code allocates and initializes an array of `N` integers, copies it to the device, launches the kernel, copies the result back, and prints a few elements to verify correct backward processing.
- Basic CUDA error checking is performed with a helper macro `CUDA_CHECK`.
- Block size is chosen as 256 threads; grid size is computed to cover all elements.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that processes elements backwards from N-1 down to 0
__global__ void processBackward(int *data, int N)
{
    int tid   = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    int stride = blockDim.x * gridDim.x;               // total stride

    int start = N - 1 - tid;                          // first index for this thread

    if (start < 0) return;                            // thread has no work

    for (int i = start; i >= 0; i -= stride) {
        // Example operation: double the element
        data[i] = 2 * data[i];
    }
}

int main(void)
{
    const int N = 1024;                  // number of elements
    size_t bytes = N * sizeof(int);

    // Host allocation and initialization
    int *h_data = (int*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;                    // simple pattern 0,1,2,...
    }

    // Device allocation
    int *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    processBackward<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    // Verify result (print first and last 10 elements)
    printf("First 10 elements after backward processing:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    printf("Last 10 elements after backward processing:\n");
    for (int i = N - 10; i < N; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return 0;
}
