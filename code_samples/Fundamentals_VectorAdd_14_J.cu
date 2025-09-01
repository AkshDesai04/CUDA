```cu
/*
Aim: Handle large `double` vectors with boundary checks.

Thinking:
The requirement is to process very large vectors of type `double` on the GPU while
ensuring that any access to the underlying arrays stays within valid bounds.
The classic approach in CUDA is to launch many threads and let each thread
process multiple elements using a grid‑stride loop. This pattern works well
for vectors whose size may exceed the total number of threads that can be
launched in a single kernel launch.

Key design decisions:
1. Use `size_t` for vector size on the host to support sizes larger than 2^32.
2. Use a constant block size (e.g., 256 threads) and compute the required
   number of blocks. The actual number of blocks is capped by the device's
   maximum grid size (retrieved via `cudaDeviceProp::maxGridSize`), but the
   grid‑stride loop guarantees all elements are processed even if the grid
   is smaller.
3. The kernel performs a simple element‑wise addition of two input vectors
   into an output vector, with a boundary check inside the stride loop
   (`if (i < N)`).
4. Error checking is performed after every CUDA API call using a helper
   macro `CHECK_CUDA`.
5. The host code accepts the vector length as a command‑line argument,
   allocates host memory, initializes the vectors with random doubles,
   copies them to device memory, launches the kernel, copies the result back,
   and finally verifies a few entries to ensure correctness.

With this structure the program can handle vectors that are orders of
magnitude larger than the GPU’s single‑kernel launch capacity while
remaining safe from out‑of‑bounds accesses.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Kernel: element-wise addition of two double vectors with boundary checks */
__global__ void add_vectors(const double *a, const double *b, double *c, size_t N)
{
    /* Compute global thread index */
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    /* Grid‑stride loop to cover all elements */
    for (size_t i = idx; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[])
{
    /* Default vector size: 16 million elements (~128 MB for double) */
    size_t N = 16 * 1024 * 1024;
    if (argc > 1) {
        N = (size_t)strtoull(argv[1], NULL, 10);
    }
    printf("Vector size: %zu\n", N);

    /* Allocate host memory */
    double *h_a = (double *)malloc(N * sizeof(double));
    double *h_b = (double *)malloc(N * sizeof(double));
    double *h_c = (double *)malloc(N * sizeof(double));
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize input vectors with random doubles */
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = (double)rand() / RAND_MAX;
        h_b[i] = (double)rand() / RAND_MAX;
    }

    /* Allocate device memory */
    double *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_a, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&d_b, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&d_c, N * sizeof(double)));

    /* Copy input data from host to device */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    const int THREADS_PER_BLOCK = 256;
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    /* Retrieve device properties to cap grid size if necessary */
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    if (blocksPerGrid > prop.maxGridSize[0]) {
        blocksPerGrid = prop.maxGridSize[0];
    }

    printf("Launching kernel with %d blocks of %d threads.\n",
           blocksPerGrid, THREADS_PER_BLOCK);

    /* Launch the kernel */
    add_vectors<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());      /* Check for kernel launch errors */
    CHECK_CUDA(cudaDeviceSynchronize()); /* Wait for kernel to finish */

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, N * sizeof(double), cudaMemcpyDeviceToHost));

    /* Simple verification: compare first few elements */
    int correct = 1;
    for (int i = 0; i < 10; ++i) {
        double expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-12) {
            printf("Mismatch at index %d: host=%f, device=%f, expected=%f\n",
                   i, h_c[i], expected, h_a[i] + h_b[i]);
            correct = 0;
            break;
        }
    }
    if (correct) {
        printf("Verification passed for first 10 elements.\n");
    } else {
        printf("Verification failed.\n");
    }

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
```