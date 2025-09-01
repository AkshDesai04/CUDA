/*
What is the limitation of `cudaMemset`? (It can only set every byte to the same value). Explain why you can't use it to initialize an array of floats to `1.0f`.

The limitation is that `cudaMemset` works on raw memory at the byte level: it writes the same 8‑bit value to every byte of the specified range. CUDA does not provide a typed memset that would write the same 32‑bit float value to each element. Because a float value is 4 bytes long, the bytes that make up a particular float can be different. For example, the IEEE‑754 representation of `1.0f` is `0x3f800000`, which in little‑endian order is the byte sequence `00 00 80 3f`. These four bytes are not all equal, so there is no single byte value that could be passed to `cudaMemset` to produce `1.0f` everywhere. The only values that can be reliably set with `cudaMemset` are those whose byte representation is identical for all four bytes, such as `0.0f` (all bytes zero). Therefore, to initialize a float array to `1.0f` you must use a kernel or other typed memory initialization routine, not `cudaMemset`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to set each float element to a given value
__global__ void setFloatKernel(float *arr, int n, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        arr[idx] = val;
}

int main(void)
{
    const int N = 8;
    size_t size = N * sizeof(float);
    float *d_arr;

    // Allocate device memory
    cudaMalloc((void **)&d_arr, size);

    /* -------------------------------------------------------------
     * Attempt to set all elements to 1.0f using cudaMemset
     * -------------------------------------------------------------
     * This will set every byte to the value 0xFF (or any value you pass).
     * The byte pattern 0xFF 0xFF 0xFF 0xFF corresponds to the float -nan,
     * not 1.0f. The result shows that cudaMemset cannot be used to set
     * non‑zero, non‑zero‑byte values for floating point types.
     */
    cudaMemset(d_arr, 0xFF, size);  // wrong for 1.0f

    float h_arr[N];
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    printf("After cudaMemset(0xFF) on float array:\n");
    for (int i = 0; i < N; ++i)
        printf("%f ", h_arr[i]);
    printf("\n");

    /* -------------------------------------------------------------
     * Correct initialization using a kernel
     * -------------------------------------------------------------
     */
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    setFloatKernel<<<blocks, threadsPerBlock>>>(d_arr, N, 1.0f);

    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    printf("After kernel setFloatKernel with 1.0f:\n");
    for (int i = 0; i < N; ++i)
        printf("%f ", h_arr[i]);
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    return 0;
}
