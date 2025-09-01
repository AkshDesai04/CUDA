/*
After allocating with `cudaMalloc`, use `cudaMemset` to set all the bytes of the device array to 0 before copying the host data over.
[Thinking]
The user wants a CUDA program that demonstrates allocating memory on the device with cudaMalloc, then immediately setting that memory to zero using cudaMemset, before copying some host data into it with cudaMemcpy.  
We will write a minimal but complete CUDA C program that does the following:
1. Define a simple kernel to verify the data on the device (e.g., sums an array).
2. In main():
   - Allocate a host array and initialize it with some values.
   - Allocate a device array with cudaMalloc.
   - Use cudaMemset to zero out the device memory.
   - Copy the host array to the device with cudaMemcpy.
   - Launch the kernel to compute a result (e.g., sum of array).
   - Copy the result back to host and print it.
   - Clean up memory.
3. Include proper error checking after each CUDA API call using a helper macro.
The code is self‑contained, uses only CUDA runtime API, and compiles as a .cu file. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

/* Simple kernel that sums an array element‑wise and writes the result to output */
__global__ void sumKernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];  // For demonstration we just copy
    }
}

int main(void) {
    const int N = 1024;
    size_t size = N * sizeof(float);

    /* Allocate and initialize host array */
    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)i;
        h_out[i] = -1.0f;  // initialize to a non‑zero value
    }

    /* Allocate device memory */
    float *d_in = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_in, size));

    /* Set all device bytes to 0 */
    CUDA_CHECK(cudaMemset(d_in, 0, size));

    /* Copy host data to device */
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    /* Launch kernel to copy data back to a separate device array */
    float *d_out = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_out, size));
    CUDA_CHECK(cudaMemset(d_out, 0, size));  // optional: clear output array

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    /* Verify result */
    int correct = 1;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != h_in[i]) {
            correct = 0;
            printf("Mismatch at index %d: host %f != device %f\n",
                   i, h_in[i], h_out[i]);
            break;
        }
    }
    if (correct) {
        printf("Success: all values match.\n");
    }

    /* Clean up */
    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
