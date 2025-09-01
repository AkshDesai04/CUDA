```cuda
/*
In a kernel with many threads, have every thread try to modify the same member of the single struct instance (e.g., `d_data->a = threadIdx.x;`). What is the final value after copying back? (Race condition).

Thinking:
The aim of this program is to illustrate a classic race condition that occurs when multiple threads concurrently write to the same memory location without any synchronization. We define a simple struct `Data` that contains a single integer member `a`. A device pointer to one instance of this struct is allocated on the GPU. The kernel `race_kernel` is launched with many threads, each of which assigns its own `threadIdx.x` value to the shared member `data->a`. Since the writes are not ordered or protected by any atomic operation or synchronization primitive, the final value of `a` after kernel execution is indeterminate – it depends on which thread writes last, which in turn depends on the scheduling and execution order determined by the GPU runtime. After copying the struct back to the host, we print the value of `a` to observe the outcome. Running the program multiple times often yields different results, demonstrating the non-deterministic nature of the race condition.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct Data {
    int a;
};

__global__ void race_kernel(Data *data) {
    // Every thread writes its threadIdx.x to the same memory location.
    data->a = threadIdx.x;
}

int main() {
    Data *d_data = nullptr;
    Data h_data{0};

    // Allocate device memory for one instance of Data
    cudaMalloc((void**)&d_data, sizeof(Data));

    // Launch kernel with many threads (e.g., 256)
    int threadsPerBlock = 256;
    race_kernel<<<1, threadsPerBlock>>>(d_data);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_data, d_data, sizeof(Data), cudaMemcpyDeviceToHost);

    // Print the final value
    printf("Final value of data.a after race: %d\n", h_data.a);

    // Clean up
    cudaFree(d_data);

    return 0;
}
```