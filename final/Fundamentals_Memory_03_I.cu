/*
Perform a three-way copy: HtoD to `d_a`, DtoD from `d_a` to `d_b`, DtoD from `d_b` to `d_c`. Verify `d_c`.

My thoughts: The program will allocate a host array, initialize it, allocate three device arrays of the same size. Then copy host array to d_a using cudaMemcpyHostToDevice. Then copy d_a to d_b with cudaMemcpyDeviceToDevice, and d_b to d_c similarly. Then copy d_c back to host for verification. We'll include error checking after each CUDA call. We'll also print out the host array and the verified array to confirm they match. The size will be a small number for demonstration, e.g., 10 elements. We'll include #include <stdio.h> and #include <cuda_runtime.h>.

We'll use a macro to check errors: #define CUDA_CHECK(call) ... etc. We'll also free device memory and exit. The main function will return 0 on success.

That covers all. We'll produce a full .cu file with proper includes, main, etc. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    const int N = 10;
    size_t size = N * sizeof(int);
    int h_a[N];
    int h_c[N];

    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 10;  // arbitrary values
    }

    /* Allocate device memory */
    int *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    /* HtoD: copy h_a to d_a */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    /* DtoD: copy d_a to d_b */
    CUDA_CHECK(cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice));

    /* DtoD: copy d_b to d_c */
    CUDA_CHECK(cudaMemcpy(d_c, d_b, size, cudaMemcpyDeviceToDevice));

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify d_c by comparing h_a and h_c */
    int ok = 1;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i]) {
            ok = 0;
            break;
        }
    }

    if (ok) {
        printf("Verification succeeded: d_c matches original data.\n");
    } else {
        printf("Verification failed: d_c does not match original data.\n");
    }

    /* Optional: print arrays */
    printf("h_a: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_a[i]);
    printf("\n");

    printf("h_c: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_c[i]);
    printf("\n");

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
