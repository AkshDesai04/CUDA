/*
Change the macro to also print the original CUDA call as a string using the `#call` preprocessor directive.

The goal is to create a debugging helper that wraps any CUDA Runtime API call, checks for an error return value, and if an error occurs, prints a helpful message that includes the file and line where the call was made, the stringified version of the call itself (using the preprocessor stringification operator `#call`), and the human‑readable CUDA error string.  This allows a developer to quickly see exactly which API invocation failed and why, without having to manually duplicate the call string in the error message.

To achieve this, we use a macro `CUDA_CHECK` that takes a single argument `call`. The macro executes the call, stores its return value, and then performs the error check.  The `#call` operator turns the argument into a string literal, which we print along with the file name (`__FILE__`), line number (`__LINE__`), and the error message returned by `cudaGetErrorString`.  We also add a `do { … } while(0)` wrapper to ensure the macro behaves like a single statement.

The rest of the file demonstrates a minimal CUDA program that allocates two vectors, copies data to the device, runs a simple element‑wise addition kernel, copies the result back, and checks every CUDA call with `CUDA_CHECK`.  The program should compile with `nvcc` and run on any device that supports CUDA.  The macro is intentionally written to be reusable for any CUDA Runtime API function. 
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Macro to check CUDA runtime API calls and print the original call string on error */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d\n", __FILE__, __LINE__);    \
            fprintf(stderr, "  Call: %s\n", #call);                          \
            fprintf(stderr, "  Error: %s (%d)\n", cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Simple kernel that adds two integer arrays element-wise */
__global__ void vector_add(const int *a, const int *b, int *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const int N = 256;
    const int size = N * sizeof(int);

    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 128;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify result */
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            fprintf(stderr, "Result mismatch at index %d: %d + %d != %d\n",
                    i, h_a[i], h_b[i], h_c[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful! All results match.\n");
    } else {
        printf("Vector addition failed.\n");
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    CUDA_CHECK(cudaDeviceReset());
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
