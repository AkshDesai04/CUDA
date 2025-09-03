/*
Define a custom exception class, e.g., `CudaException`, that inherits from `std::runtime_error` and stores the error code, error string, file, and line.

[Thinking]
To satisfy the requirement we create a header‐like comment containing the exact aim of the program.  Following the comment we provide a complete .cu source file.  The file includes the necessary CUDA and C++ headers, declares a `CudaException` class that stores an error code, the corresponding CUDA error string, the source file, and line number, and constructs a meaningful error message.  A helper macro `CHECK_CUDA` is provided to wrap CUDA runtime API calls; it throws a `CudaException` on failure.  An example `main` demonstrates throwing and catching the exception by attempting an intentionally erroneous CUDA call.
*/

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <iostream>

/* CudaException definition */
class CudaException : public std::runtime_error {
public:
    int errorCode_;
    std::string errorString_;
    std::string file_;
    int line_;

    CudaException(int code, const char* errorString, const char* file, int line)
        : std::runtime_error(buildMessage(code, errorString, file, line)),
          errorCode_(code),
          errorString_(errorString),
          file_(file),
          line_(line)
    {}

    int code() const noexcept { return errorCode_; }
    const std::string& errorStr() const noexcept { return errorString_; }
    const std::string& file() const noexcept { return file_; }
    int line() const noexcept { return line_; }

private:
    static std::string buildMessage(int code, const char* errorString,
                                    const char* file, int line) {
        std::string msg = "CUDA Error [" + std::to_string(code) + "]: ";
        msg += errorString ? errorString : "Unknown error";
        msg += " at " + std::string(file) + ":" + std::to_string(line);
        return msg;
    }
};

/* Macro to check CUDA runtime calls */
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            throw CudaException(err, cudaGetErrorString(err), __FILE__, __LINE__); \
        }                                                                    \
    } while (0)

/* Example kernel */
__global__ void dummyKernel() {
    // Do nothing
}

int main() {
    try {
        // Intentionally cause an error by requesting an invalid amount of memory
        int* d_ptr = nullptr;
        CHECK_CUDA(cudaMalloc((void**)&d_ptr, 0)); // 0 bytes is a valid call, but we can force an error by checking for failure manually
        // Actually, cudaMalloc(0) returns cudaSuccess. To generate an error, we can call an invalid API
        CHECK_CUDA(cudaMemset(d_ptr, 0, 1024)); // d_ptr is null, will cause an error

        // Launch dummy kernel (this part will not be reached if an error occurs above)
        dummyKernel<<<1, 1>>>();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    catch (const CudaException& e) {
        std::cerr << "Caught CudaException:\n";
        std::cerr << "  Code: " << e.code() << "\n";
        std::cerr << "  Message: " << e.what() << "\n";
        std::cerr << "  File: " << e.file() << "\n";
        std::cerr << "  Line: " << e.line() << "\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception& e) {
        std::cerr << "Caught std::exception: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Program completed successfully.\n";
    return EXIT_SUCCESS;
}
