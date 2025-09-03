/*
Make the class a template `template <typename T>` so it can manage any data type.
The aim of this program is to demonstrate how to create a templated CUDA class that can handle
arrays of any type.  I decided to build a simple wrapper around a device array, providing
allocation, deallocation, and host/device copy utilities.  In addition, I added a small
kernel that operates generically on the array, squaring each element.  The program
instantiates this template for several data types (int, float, double) to showcase its
generic nature.  The class uses `cudaMalloc`/`cudaFree` for memory management and
`cudaMemcpy` for data transfer.  Error checking is performed after each CUDA call
to ensure correctness.  The main function initializes host data, copies it to the
device, launches the kernel, copies results back, and prints them.  The code is
self‑contained and can be compiled with nvcc. 
*/

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__       \
                      << " (" << cudaGetErrorString(err) << ")" << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Templated class that manages a device array
template <typename T>
class DeviceArray {
private:
    T* d_ptr;          // Device pointer
    size_t size;       // Number of elements

public:
    // Constructor: allocate memory on the device
    explicit DeviceArray(size_t sz) : size(sz), d_ptr(nullptr) {
        if (size > 0) {
            CHECK_CUDA(cudaMalloc(&d_ptr, size * sizeof(T)));
        }
    }

    // Destructor: free device memory
    ~DeviceArray() {
        if (d_ptr) {
            CHECK_CUDA(cudaFree(d_ptr));
        }
    }

    // Copy data from host to device
    void set(const T* h_ptr) {
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Copy data from device to host
    void get(T* h_ptr) const {
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // Return raw device pointer (const)
    const T* data() const { return d_ptr; }
    T* data() { return d_ptr; }

    // Return number of elements
    size_t getSize() const { return size; }
};

// Kernel that squares each element of the array
template <typename T>
__global__ void squareKernel(T* arr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = arr[idx] * arr[idx];
    }
}

// Helper to launch the kernel with optimal launch parameters
template <typename T>
void launchSquareKernel(DeviceArray<T>& d_arr) {
    const size_t n = d_arr.getSize();
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr.data(), n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

int main() {
    // Example with int
    const size_t N = 10;
    int h_int[N];
    for (size_t i = 0; i < N; ++i) h_int[i] = static_cast<int>(i + 1);

    DeviceArray<int> d_int(N);
    d_int.set(h_int);

    launchSquareKernel(d_int);

    int h_result_int[N];
    d_int.get(h_result_int);

    std::cout << "Int results:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_int[i] << "^2 = " << h_result_int[i] << std::endl;
    }

    // Example with float
    float h_float[N];
    for (size_t i = 0; i < N; ++i) h_float[i] = static_cast<float>(i) + 0.5f;

    DeviceArray<float> d_float(N);
    d_float.set(h_float);

    launchSquareKernel(d_float);

    float h_result_float[N];
    d_float.get(h_result_float);

    std::cout << "\nFloat results:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_float[i] << "^2 = " << h_result_float[i] << std::endl;
    }

    // Example with double
    double h_double[N];
    for (size_t i = 0; i < N; ++i) h_double[i] = static_cast<double>(i) + 1.0;

    DeviceArray<double> d_double(N);
    d_double.set(h_double);

    launchSquareKernel(d_double);

    double h_result_double[N];
    d_double.get(h_result_double);

    std::cout << "\nDouble results:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_double[i] << "^2 = " << h_result_double[i] << std::endl;
    }

    return 0;
}
