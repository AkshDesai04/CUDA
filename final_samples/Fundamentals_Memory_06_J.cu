/*
 * Aim of the program:
 * Write a comment block with best practices for avoiding memory leaks in CUDA (e.g., using RAII wrappers).
 *
 * My thinking:
 * 1. The user requested a comment block that lists best practices for avoiding memory leaks in CUDA.
 * 2. The developer instruction requires that I output a .cu file, with the aim of the program written exactly as given in the prompt.
 * 3. Therefore, the first thing I will put in the file is a multi-line comment that includes the exact aim string.
 * 4. Inside that same comment, I will also add a detailed explanation of best practices, including:
 *    - Always check return codes from CUDA API calls.
 *    - Use RAII wrappers such as unique_ptr with custom deleters or custom smart pointer classes that call cudaFree.
 *    - Prefer C++ standard containers for host data.
 *    - Avoid global device pointers; keep allocations scoped.
 *    - Use cudaMallocManaged carefully.
 *    - Ensure cudaDeviceSynchronize or cudaMemcpy before freeing memory.
 * 5. After the comment, I will write a minimal CUDA program that demonstrates the use of a simple RAII wrapper class (DeviceArray) that manages device memory.
 * 6. The program will include a trivial kernel to illustrate that allocations are properly freed when the object goes out of scope.
 * 7. I will keep the code short, self-contained, and compile-able with nvcc.
 * 8. Finally, I will only output the .cu file content, nothing else.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdexcept>

/* Best practices for avoiding memory leaks in CUDA:
 * 1. Check return status of every CUDA API call. Use a helper macro or function to throw on error.
 * 2. Use RAII wrappers for device memory. E.g., a unique_ptr with a custom deleter that calls cudaFree.
 * 3. Prefer std::unique_ptr<DeviceArray<T>> for device buffers; this ensures automatic cleanup.
 * 4. Keep device pointers local to functions or classes; avoid global device pointers.
 * 5. Synchronize (cudaDeviceSynchronize) before freeing memory if pending kernel launches exist.
 * 6. For managed memory (cudaMallocManaged), use cudaDeviceSynchronize or cudaMemcpyAsync to ensure all accesses complete before freeing.
 * 7. Avoid mixing C and C++ memory management; stick to either CUDA APIs or RAII, but not both for the same resource.
 * 8. Wrap kernels in error-checking wrappers that validate launch success and check for launch failures.
 * 9. Use constexpr or inline functions to avoid unnecessary heap allocations.
 * 10. For large projects, consider using existing libraries like thrust::device_vector which already implement RAII for device memory.
 */

/* Helper macro to check CUDA API calls */
#define CUDA_CHECK(call)                                              \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error");                  \
        }                                                            \
    } while (0)

/* RAII wrapper for device memory */
template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t n) : size_(n) {
        CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
    }
    ~DeviceArray() {
        if (data_) {
            // Ensure all work on this memory is complete before freeing
            cudaDeviceSynchronize();
            cudaFree(data_);
        }
    }
    // Delete copy semantics
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    // Allow move semantics
    DeviceArray(DeviceArray&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (data_) {
                cudaDeviceSynchronize();
                cudaFree(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    T* data() const { return data_; }
    size_t size() const { return size_; }
private:
    T* data_ = nullptr;
    size_t size_;
};

/* Simple kernel that sets each element to its index */
__global__ void init_kernel(int* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}

int main() {
    const int N = 1024;
    try {
        DeviceArray<int> dev_arr(N);

        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel<<<blocks, threadsPerBlock>>>(dev_arr.data(), N);
        CUDA_CHECK(cudaGetLastError()); // Check for launch errors

        // Copy back to host to verify
        std::unique_ptr<int[]> host_arr(new int[N]);
        CUDA_CHECK(cudaMemcpy(host_arr.get(), dev_arr.data(), N * sizeof(int), cudaMemcpyDeviceToHost));

        // Simple validation
        for (int i = 0; i < N; ++i) {
            if (host_arr[i] != i) {
                std::cerr << "Mismatch at " << i << ": got " << host_arr[i] << std::endl;
                return EXIT_FAILURE;
            }
        }
        std::cout << "All values initialized correctly." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
