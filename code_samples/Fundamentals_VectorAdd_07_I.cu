```cpp
/*
Aim: Implement `output[i] = sinf(input[i])`.

Thinking:
The task is to create a simple CUDA program that computes the sine of each element
in an input array and writes the result to an output array.  The implementation
needs to be a complete, compilable .cu file that uses the CUDA runtime API.
Here is the plan:

1. **Kernel design**  
   - A straightforward element‑wise kernel that receives a pointer to the input
     array, a pointer to the output array, and the number of elements `N`.  
   - Each thread will compute one output element:  
     `output[idx] = sinf(input[idx]);`  
     where `idx` is calculated from the block and thread indices.  
   - Boundary checking (`idx < N`) is required to avoid out‑of‑bounds memory
     access when the grid size is not an exact multiple of the block size.

2. **Host code**  
   - Allocate host arrays for input and output.  
   - Initialize the input array with some test values (e.g., a linear space or
     a simple pattern).  
   - Allocate device memory for both arrays.  
   - Copy the input array from host to device.  
   - Launch the kernel with a suitable grid and block configuration.  
   - Copy the output array back to the host.  
   - Optionally print the first few results for verification.  
   - Free both host and device memory.

3. **Error handling**  
   - Use the `cudaGetLastError()` and `cudaDeviceSynchronize()` calls to
     capture and report any kernel launch or runtime errors.  
   - Wrap common CUDA calls in a macro or inline function to make the code
     cleaner.

4. **Compilation**  
   - The file is a standard CUDA source file (.cu) and can be compiled with  
     `nvcc -o sin_example sin_example.cu`.

All of the above considerations are reflected in the code below.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",  \
                    __FILE__, __LINE__, err, cudaGetErrorString(err),     \
                    #call);                                               \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Kernel that computes output[i] = sinf(input[i])
__global__ void sin_kernel(const float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = sinf(input[idx]);
    }
}

int main(void)
{
    const int N = 1 << 20;          // Number of elements (e.g., 1M)
    const size_t size = N * sizeof(float);

    // Host allocations
    float *h_input  = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input with a simple pattern (e.g., 0, 0.1, 0.2, ...)
    for (int i = 0; i < N; ++i) {
        h_input[i] = 0.1f * (float)i;
    }

    // Device allocations
    float *d_input  = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    sin_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of sinf(input):\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("sinf(%.3f) = %.6f\n", h_input[i], h_output[i]);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}
```