```c
/*
What happens if you mismatch types? E.g., device memory is `double*` but the kernel signature is `float*`?

Thinking:
- In CUDA, the type of the pointer parameter in the kernel must match the type of the pointer you pass.
- If you compile without an explicit cast, the compiler will reject the code with a type mismatch error:
    error: cannot convert 'double*' to 'float*' in argument passing
- You can bypass the compile-time check by explicitly casting the pointer, e.g. reinterpret_cast<float*>(devDouble).
- However, this introduces a logical mismatch:
    • A double occupies 8 bytes; a float occupies 4 bytes.
    • Treating a double* as a float* means each float element refers to only half of a double element.
- When the kernel writes to a float*, it will only affect the lower 4 bytes of each 8‑byte double.
- The upper 4 bytes of the double remain unchanged, leading to corrupted data.
- The kernel may still execute, but the results will be incorrect.
- If the kernel accesses memory out of bounds due to incorrect size assumptions, it may cause a segmentation fault or memory corruption.
- Therefore, it is crucial to ensure that the host allocation, device allocation, and kernel signature all agree on the data type.

This example demonstrates a deliberate mismatch: host and device memory are allocated as doubles,
but the kernel receives a float* cast. The kernel adds 1.0f to each float element, which only
modifies the lower half of each double, producing unexpected results.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Add 1.0f to each float element (affects only the lower 4 bytes of each double)
        data[idx] += 1.0f;
    }
}

int main()
{
    const int N = 4;
    double hostDouble[N] = {1.0, 2.0, 3.0, 4.0};

    double* devDouble = nullptr;
    cudaMalloc((void**)&devDouble, N * sizeof(double));
    cudaMemcpy(devDouble, hostDouble, N * sizeof(double), cudaMemcpyHostToDevice);

    // Intentionally cast double* to float* and launch the kernel
    kernel<<<1, N>>>(reinterpret_cast<float*>(devDouble), N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    else
    {
        // Copy back the potentially corrupted data
        cudaMemcpy(hostDouble, devDouble, N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i)
        {
            printf("hostDouble[%d] = %lf\n", i, hostDouble[i]);
        }
    }

    cudaFree(devDouble);
    return 0;
}
```