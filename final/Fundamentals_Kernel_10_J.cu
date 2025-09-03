```c
/* 
Aim: 
Pass the struct by constant reference (`const MyData&`) to the kernel. This can be more efficient for large structs.

Thinking:
The goal is to demonstrate how a large struct can be passed to a CUDA kernel by constant reference instead of by value, which can reduce unnecessary copies in some scenarios. 
In CUDA, kernel arguments are marshaled by the runtime. If we declare a kernel parameter as a reference (`const MyData&`), the compiler will treat it as a pointer internally and avoid copying the entire struct into the kernel's argument area. 
We define a struct `MyData` that contains a large array to illustrate this. The kernel `process` takes a `const MyData&` and writes twice each element of the array into an output buffer. The host code initializes the struct, launches the kernel, and copies the result back to the host for verification. 
The code is self-contained and can be compiled with `nvcc`. 
*/

#include <cstdio>
#include <cuda_runtime.h>

// Large struct with many elements
struct MyData {
    int n;                    // number of valid elements in arr
    float arr[1024];          // large array
};

// Kernel that processes the struct by const reference
__global__ void process(const MyData& data, float* output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data.n) {
        // For demonstration, double the value
        output[idx] = data.arr[idx] * 2.0f;
    }
}

int main()
{
    // Host data initialization
    MyData hData;
    hData.n = 1024;
    for (int i = 0; i < hData.n; ++i) {
        hData.arr[i] = static_cast<float>(i);
    }

    // Device output array
    float *dOutput;
    cudaMalloc((void**)&dOutput, hData.n * sizeof(float));

    // Launch kernel with a const reference to hData
    const MyData& dataRef = hData;                // const reference to the struct
    dim3 blockSize(256);
    dim3 gridSize((hData.n + blockSize.x - 1) / blockSize.x);
    process<<<gridSize, blockSize>>>(dataRef, dOutput);

    // Copy result back to host
    float *hOutput = new float[hData.n];
    cudaMemcpy(hOutput, dOutput, hData.n * sizeof(float), cudaMemcpyDeviceToHost);

    // Simple verification
    bool ok = true;
    for (int i = 0; i < hData.n; ++i) {
        if (hOutput[i] != hData.arr[i] * 2.0f) {
            ok = false;
            printf("Mismatch at index %d: expected %f, got %f\n", i, hData.arr[i]*2.0f, hOutput[i]);
            break;
        }
    }
    printf("Result %s\n", ok ? "OK" : "FAIL");

    // Clean up
    delete[] hOutput;
    cudaFree(dOutput);

    return 0;
}
```