```cuda
/* 
Aim of the program:
Allocate `d_a` and `d_b`. Use `cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice)` to initialize `d_a`. 
What is the error? Change `h_a` to `d_a`. What is the error? Explain why the kind parameter is critical.

Thinking:
The program demonstrates two common mistakes when using cudaMemcpy. 
1) Copying from a host pointer (h_a) to a device pointer (d_b) while intending to initialize d_a. 
   This results in d_b being incorrectly set while d_a remains uninitialized, leading to logic errors or 
   undefined results when d_a is later used. The copy operation succeeds but the data ends up in the wrong 
   memory region, so subsequent device computations will use garbage values. 
2) Swapping the source to d_a but keeping the kind as cudaMemcpyHostToDevice. Since d_a is a device pointer, 
   cudaMemcpyHostToDevice expects a host source pointer, which triggers an invalid argument error (error code 11). 
   The kernel or API will report that the source or destination addresses are invalid for the requested transfer 
   kind. 

The cudaMemcpyKind parameter tells the runtime the direction of the copy: host-to-device, device-to-host, 
device-to-device, or default (which infers based on pointer addresses). The runtime checks that the source and 
destination pointers match the expected kind; if they don't, it aborts the copy with an error. This is critical 
to prevent accidental data corruption or memory violations, as the CUDA memory model treats host and device 
memory spaces separately. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 5
#define CHECK(call)                                                   \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main(void)
{
    /* Host array */
    int h_a[N] = {1, 2, 3, 4, 5};

    /* Device pointers */
    int *d_a = NULL;
    int *d_b = NULL;
    size_t size = N * sizeof(int);

    /* Allocate device memory */
    CHECK(cudaMalloc((void**)&d_a, size));
    CHECK(cudaMalloc((void**)&d_b, size));

    /* ----------- Mistake 1: Wrong destination in cudaMemcpy ----------- */
    printf("Attempting to copy h_a to d_b but intending to init d_a...\n");
    cudaError_t err1 = cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice);
    if (err1 != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err1));
    else
        printf("Copy succeeded, but d_a remains uninitialized.\n");

    /* Read back d_a to show its content (should be uninitialized) */
    int h_check[N] = {0};
    CHECK(cudaMemcpy(h_check, d_a, size, cudaMemcpyDeviceToHost));
    printf("Contents of d_a after mistaken copy: ");
    for (int i = 0; i < N; i++)
        printf("%d ", h_check[i]);
    printf("\n");

    /* ----------- Mistake 2: Wrong kind parameter ----------- */
    printf("\nAttempting to copy d_a to d_b with HostToDevice kind...\n");
    cudaError_t err2 = cudaMemcpy(d_b, d_a, size, cudaMemcpyHostToDevice);
    if (err2 != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err2));
    else
        printf("Copy succeeded (unexpected).\n");

    /* Clean up */
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    return 0;
}
```