/*
Aim: Modify your `CHECK` macro to not exit, but instead throw a C++ exception. This allows for more sophisticated error handling in larger applications.

Thinking:
- The original CHECK macro was used to verify the return status of CUDA API calls. It typically printed an error message and called exit(1) on failure.
- In larger applications, abruptly terminating the program is undesirable; instead, we want to propagate the error upwards via C++ exception handling.
- We'll replace the exit call with throwing std::runtime_error, including contextual information (file, line, and error string).
- The macro will be used for both synchronous API calls and for checking the result of a kernel launch (via cudaGetLastError).
- We will also create a small example that allocates memory, copies data, launches a kernel, and uses the CHECK macro to demonstrate error handling.
- Exceptions are caught in main and printed, allowing the program to exit gracefully after cleanup.
- The code is written as a single .cu file, ready to compile with nvcc and requires C++11 support for std::to_string.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

/* 
 * Macro to check the status of a CUDA API call.
 * If the call fails, throws a std::runtime_error with file, line, and error message.
 */
#define CHECK(call)                                                           \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(                                         \
                std::string("CUDA error in file ") + __FILE__ +              \
                " line " + std::to_string(__LINE__) + ": " +                \
                cudaGetErrorString(err));                                    \
        }                                                                     \
    } while (0)

/*
 * Simple kernel that adds two arrays element-wise.
 */
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

/*
 * Helper function to initialize host arrays.
 */
void initArray(float *arr, int n, float value)
{
    for (int i = 0; i < n; ++i)
        arr[i] = value;
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    initArray(h_a, N, 1.0f);
    initArray(h_b, N, 2.0f);

    // Allocate device memory
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    try {
        CHECK(cudaMalloc((void**)&d_a, size));
        CHECK(cudaMalloc((void**)&d_b, size));
        CHECK(cudaMalloc((void**)&d_c, size));

        // Copy data from host to device
        CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

        // Check for kernel launch errors
        CHECK(cudaGetLastError());

        // Copy result back to host
        CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

        // Verify result
        bool success = true;
        for (int i = 0; i < N; ++i) {
            if (h_c[i] != 3.0f) {
                success = false;
                std::cerr << "Verification failed at index " << i
                          << ": " << h_c[i] << " != 3.0f\n";
                break;
            }
        }
        if (success) {
            std::cout << "Vector addition successful. All results are 3.0f.\n";
        } else {
            std::cerr << "Vector addition failed.\n";
        }

    } catch (const std::runtime_error &e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Caught unknown exception." << std::endl;
    }

    // Clean up device memory
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);

    // Clean up host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
