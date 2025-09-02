```cpp
/* 
Aim of the program: 
Define a struct `MyData { int a; float b; }`. In the host, initialize it with `{10, 3.14f}`. Pass it to the kernel and have the kernel print both members. 

Thinking: 
- The struct must be defined in a way that is accessible from both host and device. Since it's a simple POD type, we can simply declare it in the global scope and let the compiler handle the layout.
- On the host, we create an instance of `MyData` and set `a = 10` and `b = 3.14f`. 
- To send this data to the kernel, we need a device copy. We'll allocate memory on the device using `cudaMalloc`, copy the struct over with `cudaMemcpy`, and then launch a kernel that receives a pointer to the struct. 
- Inside the kernel we can use the device `printf` function to output the two members. CUDA device printf requires the device to support compute capability >= 2.0, and it prints to the standard output on the host. 
- After kernel execution we call `cudaDeviceSynchronize()` to ensure all prints are flushed. 
- Simple error checking is added for CUDA API calls so that failures are reported. 
- The program is self‑contained and can be compiled with `nvcc -arch=sm_35 -o myprog myprog.cu`. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Define the struct that will be shared between host and device */
struct MyData
{
    int a;
    float b;
};

/* Kernel that prints the members of MyData */
__global__ void printMyData(const MyData* data)
{
    // Each thread could print, but we only launch one thread here
    printf("Device: a = %d, b = %f\n", data->a, data->b);
}

/* Utility macro for checking CUDA errors */
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main()
{
    /* Host-side data initialization */
    MyData h_data = {10, 3.14f};

    /* Device memory allocation */
    MyData* d_data = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_data, sizeof(MyData)));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_data, &h_data, sizeof(MyData), cudaMemcpyHostToDevice));

    /* Launch kernel with a single thread */
    printMyData<<<1, 1>>>(d_data);

    /* Ensure all prints are completed */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Clean up */
    CHECK_CUDA(cudaFree(d_data));

    printf("Host: Finished.\n");
    return 0;
}
```