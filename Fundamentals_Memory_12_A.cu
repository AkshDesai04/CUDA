/*
Add a copy constructor and a copy assignment operator that perform a deep copy (allocating new device memory and doing a `cudaMemcpyDeviceToDevice`).

I started by determining the minimal CUDA program that demonstrates a class owning device memory and the need for a deep copy. The class, CudaArray, will:
1. Hold a pointer to device memory (float*) and a size.
2. Allocate memory in the constructor, free it in the destructor.
3. Provide a method to copy data from host to device.
4. Provide a method to copy data from device back to host for printing.
5. Include a kernel to perform a simple operation (doubling each element) to verify that data is on the device.
6. Add a copy constructor that:
   - Allocates new device memory with cudaMalloc of the same size.
   - Copies the existing device data to the new memory with cudaMemcpyDeviceToDevice.
7. Add a copy assignment operator that:
   - Checks for self-assignment.
   - Frees existing memory.
   - Performs the same allocation and device-to-device copy as the copy constructor.
   - Handles errors appropriately.
8. In main, I will:
   - Create a CudaArray, initialize host data, copy to device.
   - Launch the kernel.
   - Copy data back to host and print.
   - Use the copy constructor to create a new CudaArray from the first.
   - Use the copy assignment operator to copy into a third instance.
   - Print data from all instances to confirm the deep copy.
9. Add a simple error-checking macro to wrap CUDA calls for clarity.
10. Ensure the code compiles as a .cu file and uses only necessary includes.

The resulting code demonstrates deep copying of device memory in a C++ class with CUDA, fulfilling the prompt's requirement.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Simple kernel to double each element
__global__ void double_elements(float *data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}

class CudaArray {
private:
    float *d_data_;
    size_t size_;

public:
    // Constructor: allocate device memory
    CudaArray(size_t size) : d_data_(nullptr), size_(size) {
        CUDA_CHECK(cudaMalloc(&d_data_, size_ * sizeof(float)));
    }

    // Destructor: free device memory
    ~CudaArray() {
        if (d_data_) {
            CUDA_CHECK(cudaFree(d_data_));
        }
    }

    // Copy constructor: deep copy
    CudaArray(const CudaArray &other) : d_data_(nullptr), size_(other.size_) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_data_, size_ * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_data_, other.d_data_, size_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }
    }

    // Copy assignment operator: deep copy
    CudaArray &operator=(const CudaArray &other) {
        if (this == &other) return *this; // Self-assignment check

        // Free current resources
        if (d_data_) {
            CUDA_CHECK(cudaFree(d_data_));
        }

        size_ = other.size_;
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_data_, size_ * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_data_, other.d_data_, size_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        } else {
            d_data_ = nullptr;
        }
        return *this;
    }

    // Disable move semantics for simplicity
    CudaArray(CudaArray &&) = delete;
    CudaArray &operator=(CudaArray &&) = delete;

    // Copy data from host to device
    void copyFromHost(const float *h_data) {
        CUDA_CHECK(cudaMemcpy(d_data_, h_data, size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Copy data from device to host
    void copyToHost(float *h_data) const {
        CUDA_CHECK(cudaMemcpy(h_data, d_data_, size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // Launch kernel to double elements
    void doubleElements() const {
        const int threadsPerBlock = 256;
        int blocksPerGrid = (int)((size_ + threadsPerBlock - 1) / threadsPerBlock);
        double_elements<<<blocksPerGrid, threadsPerBlock>>>(d_data_, size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Getter for size
    size_t size() const { return size_; }
};

int main() {
    const size_t N = 10;
    float h_input[N];
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i + 1);
    }

    // Original array
    CudaArray array1(N);
    array1.copyFromHost(h_input);
    array1.doubleElements();

    float h_output[N];
    array1.copyToHost(h_output);
    printf("After doubling (array1):\n");
    for (size_t i = 0; i < N; ++i) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");

    // Copy constructor
    CudaArray array2 = array1;
    array2.doubleElements(); // Further double
    array2.copyToHost(h_output);
    printf("After additional doubling (array2):\n");
    for (size_t i = 0; i < N; ++i) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");

    // Copy assignment
    CudaArray array3(N); // Initially empty
    array3 = array1;
    array3.doubleElements(); // Double again
    array3.copyToHost(h_output);
    printf("After doubling via assignment (array3):\n");
    for (size_t i = 0; i < N; ++i) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");

    return 0;
}
