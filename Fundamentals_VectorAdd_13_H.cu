```cpp
/*
 * Aim of the program: Implement a clamp function: C[i] = min(max(A[i], min_val), max_val).
 *
 * Thinking process:
 *
 * 1. The requirement is to create a CUDA program in C that performs a clamp operation on an array.
 * 2. A clamp operation takes each element A[i] and constrains it within the bounds [min_val, max_val].
 *    The mathematical expression is:
 *      C[i] = min(max(A[i], min_val), max_val)
 * 3. We'll write a CUDA kernel that performs this computation for each element in parallel.
 * 4. On the host side, we will:
 *    - Allocate and initialize a sample input array A.
 *    - Define min_val and max_val.
 *    - Allocate device memory for A and C.
 *    - Copy A to device.
 *    - Launch the kernel with enough threads to cover all elements.
 *    - Copy the result C back to host.
 *    - Print the input and clamped output for verification.
 * 5. Error checking for CUDA API calls is included for robustness.
 * 6. The program is self‑contained and can be compiled with `nvcc clamp.cu -o clamp`.
 * 7. The output demonstrates the clamp operation on a small example array.
 *
 * The code below follows this plan, using `float` for simplicity.
 */

#include <stdio.h>
#include <cuda.h>

/* CUDA kernel to clamp each element of array A into [min_val, max_val] */
__global__ void clamp_kernel(const float *A, float *C, float min_val, float max_val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = A[idx];
        /* Apply clamp: C[idx] = min(max(val, min_val), max_val) */
        float clamped = val;
        if (clamped < min_val) clamped = min_val;
        if (clamped > max_val) clamped = max_val;
        C[idx] = clamped;
    }
}

/* Utility function to check CUDA errors */
void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    /* Example data */
    const int N = 10;
    float h_A[N];
    float h_C[N];
    /* Initialize input array with values ranging from -5 to 14 */
    for (int i = 0; i < N; ++i) {
        h_A[i] = -5.0f + i * 1.5f;  // Example values: -5, -3.5, -2, -0.5, 1, 2.5, 4, 5.5, 7, 8.5
    }
    float min_val = 0.0f;
    float max_val = 5.0f;

    /* Device memory pointers */
    float *d_A = NULL;
    float *d_C = NULL;

    /* Allocate device memory */
    checkCuda(cudaMalloc((void**)&d_A, N * sizeof(float)), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void**)&d_C, N * sizeof(float)), "cudaMalloc d_C");

    /* Copy input array from host to device */
    checkCuda(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy HtoD");

    /* Kernel launch parameters */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch clamp kernel */
    clamp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, min_val, max_val, N);

    /* Check for kernel launch errors */
    checkCuda(cudaGetLastError(), "Kernel launch");

    /* Copy result back to host */
    checkCuda(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpyDtoH");

    /* Print results */
    printf("Input array A:\n");
    for (int i = 0; i < N; ++i) {
        printf("%8.3f ", h_A[i]);
    }
    printf("\nClamped array C (min=%f, max=%f):\n", min_val, max_val);
    for (int i = 0; i < N; ++i) {
        printf("%8.3f ", h_C[i]);
    }
    printf("\n");

    /* Free device memory */
    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}
```