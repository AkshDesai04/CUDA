```cpp
/* 
Aim of the program: Add a copy constructor and a copy assignment operator that perform a deep copy (allocating new device memory and doing a cudaMemcpyDeviceToDevice).

Thinking:
The task is to demonstrate how to implement a copy constructor and a copy assignment operator for a class that manages device memory in CUDA. 
We will create a simple class `DeviceVector` that owns a pointer to device memory and knows its size. 
The constructor allocates device memory using `cudaMalloc`. 
The destructor frees that memory with `cudaFree`. 
The copy constructor will allocate new memory of the same size and copy the contents from the source device pointer using `cudaMemcpy` with the `cudaMemcpyDeviceToDevice` flag. 
The copy assignment operator will first check for self‑assignment, free any existing memory, allocate new memory, and copy from the source. 
We also provide a simple kernel to fill the vector with sequential values and another kernel to print the contents for verification. 
The `main` function tests the deep copy by creating an instance, copying it, modifying the copy, and then printing both original and copy to show that they are independent. 
All CUDA API calls are wrapped in a macro for error checking to simplify debugging. 
The code is fully self‑contained and can be compiled with `nvcc` as a `.cu` file. 
*/

#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel to initialize device array with sequential values
__global__ void initKernel(int* d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_arr[idx] = idx;
}

// Kernel to print device array (host-side copy)
__global__ void printKernel(const int* d_arr, int n, const char* label) {
    // Only thread 0 prints
    if (threadIdx.x == 0) {
        printf("%s: ", label);
        for (int i = 0; i < n; ++i) {
            printf("%d ", d_arr[i]);
        }
        printf("\n");
    }
}

// Class that owns a device array
class DeviceVector {
public:
    // Default constructor
    DeviceVector() : d_ptr(nullptr), size(0) {}

    // Constructor that allocates device memory and initializes it
    DeviceVector(int n) : size(n) {
        CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(int)));
        // Initialize with zeros
        CUDA_CHECK(cudaMemset(d_ptr, 0, size * sizeof(int)));
    }

    // Destructor
    ~DeviceVector() {
        if (d_ptr) {
            CUDA_CHECK(cudaFree(d_ptr));
        }
    }

    // Copy constructor: deep copy
    DeviceVector(const DeviceVector& other) : size(other.size) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_ptr, other.d_ptr, size * sizeof(int), cudaMemcpyDeviceToDevice));
        } else {
            d_ptr = nullptr;
        }
    }

    // Copy assignment operator: deep copy
    DeviceVector& operator=(const DeviceVector& other) {
        if (this == &other) return *this; // self‑assignment guard

        // Free existing memory
        if (d_ptr) {
            CUDA_CHECK(cudaFree(d_ptr));
        }

        size = other.size;
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_ptr, other.d_ptr, size * sizeof(int), cudaMemcpyDeviceToDevice));
        } else {
            d_ptr = nullptr;
        }
        return *this;
    }

    // Move constructor (optional but good practice)
    DeviceVector(DeviceVector&& other) noexcept : d_ptr(other.d_ptr), size(other.size) {
        other.d_ptr = nullptr;
        other.size = 0;
    }

    // Move assignment operator
    DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            if (d_ptr) {
                CUDA_CHECK(cudaFree(d_ptr));
            }
            d_ptr = other.d_ptr;
            size = other.size;
            other.d_ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }

    // Fill the device array with sequential values using kernel
    void fillSequential() {
        if (size == 0) return;
        int threadsPerBlock = 256;
        int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        initKernel<<<blocks, threadsPerBlock>>>(d_ptr, size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Print the device array to host
    void print(const char* label) const {
        if (size == 0) {
            printf("%s is empty.\n", label);
            return;
        }
        int threadsPerBlock = 256;
        int blocks = 1; // only need one block for printing
        printKernel<<<blocks, threadsPerBlock>>>(d_ptr, size, label);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Expose device pointer for advanced use (const)
    const int* devicePtr() const { return d_ptr; }

private:
    int* d_ptr; // pointer to device memory
    int size;   // number of elements
};

int main() {
    // Create a DeviceVector of size 10 and fill it
    DeviceVector vecA(10);
    vecA.fillSequential();
    vecA.print("vecA original");

    // Copy construct vecB from vecA
    DeviceVector vecB = vecA;
    vecB.print("vecB after copy");

    // Modify vecB
    // For demonstration, we use a simple kernel to set all elements to -1
    int threadsPerBlock = 256;
    int blocks = 1;
    initKernel<<<blocks, threadsPerBlock>>>(vecB.devicePtr(), 10);
    CUDA_CHECK(cudaDeviceSynchronize());
    vecB.print("vecB after modification");

    // Show that vecA remains unchanged
    vecA.print("vecA after vecB modification");

    // Assign vecA to a new vector via copy assignment
    DeviceVector vecC;
    vecC = vecA;
    vecC.print("vecC after assignment");

    // Clean up is automatic via destructors
    return 0;
}
```