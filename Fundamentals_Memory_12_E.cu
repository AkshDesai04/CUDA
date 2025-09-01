```cuda
/* 
Aim of the program:
Make the class a template `template <typename T>` so it can manage any data type.  

Thinking behind the design:
1. We want a simple, self‑contained example that demonstrates how a templated C++ class can be used
   in a CUDA program to manage data on both host and device.  
2. The class `CudaArray<T>` will:
   - Store a pointer to host data (`h_data`) and device data (`d_data`).
   - Keep track of the number of elements `size_`.
   - Provide constructors to allocate memory, a destructor to free it,
     and member functions to copy data between host and device.
3. For demonstration purposes we implement a small kernel
   that adds a scalar value to each element of the array.
4. In `main()` we instantiate the class twice: once with `int` and once with `float`,
   perform the kernel launch, copy the result back, and print a few elements.
5. The program uses only standard CUDA runtime API (`cudaMalloc`, `cudaMemcpy`, etc.).
   All host code is plain C++ and compiles with `nvcc`.  
6. Error checking is performed via a helper macro `CUDA_CHECK`.  
7. The code is fully self‑contained; no external headers or libraries are required.  
8. Comments inside the code explain each part of the logic for clarity.  
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

/* Templated class that manages an array of type T on host and device */
template <typename T>
class CudaArray {
public:
    /* Constructors */
    CudaArray() : h_data(nullptr), d_data(nullptr), size_(0) {}

    /* Allocate host and device memory for given number of elements */
    explicit CudaArray(size_t size) : h_data(nullptr), d_data(nullptr), size_(size) {
        h_data = new T[size];
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
    }

    /* Destructor frees host and device memory */
    ~CudaArray() {
        delete[] h_data;
        if (d_data) CUDA_CHECK(cudaFree(d_data));
    }

    /* Copy data from host to device */
    void copyToDevice() {
        if (h_data && d_data) {
            CUDA_CHECK(cudaMemcpy(d_data, h_data, size_ * sizeof(T),
                                 cudaMemcpyHostToDevice));
        }
    }

    /* Copy data from device to host */
    void copyToHost() {
        if (h_data && d_data) {
            CUDA_CHECK(cudaMemcpy(h_data, d_data, size_ * sizeof(T),
                                 cudaMemcpyDeviceToHost));
        }
    }

    /* Getters */
    T* hostPtr() { return h_data; }
    const T* hostPtr() const { return h_data; }
    T* devicePtr() { return d_data; }
    const T* devicePtr() const { return d_data; }
    size_t size() const { return size_; }

private:
    T* h_data;     /* Host pointer */
    T* d_data;     /* Device pointer */
    size_t size_;  /* Number of elements */
};

/* Kernel that adds a scalar value to each element of the array */
template <typename T>
__global__ void addScalarKernel(T* data, T scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += scalar;
    }
}

/* Launch helper for the templated kernel */
template <typename T>
void launchAddScalar(CudaArray<T>& arr, T scalar) {
    const size_t threadsPerBlock = 256;
    const size_t blocks = (arr.size() + threadsPerBlock - 1) / threadsPerBlock;
    addScalarKernel<T><<<blocks, threadsPerBlock>>>(arr.devicePtr(), scalar, arr.size());
    CUDA_CHECK(cudaGetLastError());
}

int main() {
    const size_t N = 10;

    /* Example with int */
    CudaArray<int> intArr(N);
    for (size_t i = 0; i < N; ++i) intArr.hostPtr()[i] = static_cast<int>(i);
    intArr.copyToDevice();
    launchAddScalar(intArr, 5);      // add 5 to each element
    intArr.copyToHost();
    printf("int array after adding 5:\n");
    for (size_t i = 0; i < N; ++i) printf("%d ", intArr.hostPtr()[i]);
    printf("\n");

    /* Example with float */
    CudaArray<float> floatArr(N);
    for (size_t i = 0; i < N; ++i) floatArr.hostPtr()[i] = static_cast<float>(i) * 1.1f;
    floatArr.copyToDevice();
    launchAddScalar(floatArr, 2.5f); // add 2.5 to each element
    floatArr.copyToHost();
    printf("float array after adding 2.5:\n");
    for (size_t i = 0; i < N; ++i) printf("%.2f ", floatArr.hostPtr()[i]);
    printf("\n");

    return 0;
}
```