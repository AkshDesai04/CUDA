```cpp
/*
Aim of the program: Define a custom exception class, e.g., CudaException, that inherits from std::runtime_error and stores the error code, error string, file, and line.

Thinking:
- The program is a CUDA C++ (.cu) file, but exception handling is a host-only feature in CUDA.
- The custom exception class should be derived from std::runtime_error for compatibility with standard exception handling.
- It should store:
  * an integer error code (e.g., cudaError_t)
  * a string representation of the error (from cudaGetErrorString)
  * the source file name where the exception was thrown
  * the line number where the exception was thrown
- The constructor will take these parameters and construct a descriptive message for the std::runtime_error base.
- Provide accessor methods to retrieve the stored information if needed.
- Since CUDA errors are often captured via cudaError_t, the class will accept this type as error code.
- A helper macro could be defined to automatically capture file and line, but the core requirement is just the class definition.
- No device code is needed; the class is purely host side.
- The .cu file will include <cuda_runtime.h>, <stdexcept>, <string>, and possibly <sstream> for message formatting.

Now implementing the class.
*/

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

// Custom CUDA exception class
class CudaException : public std::runtime_error {
public:
    // Constructor that accepts error code, error string, file and line
    CudaException(cudaError_t errCode,
                  const char* errStr,
                  const char* file,
                  int line)
        : std::runtime_error(buildMessage(errCode, errStr, file, line)),
          errorCode(errCode),
          errorString(errStr),
          fileName(file),
          lineNumber(line)
    {}

    // Accessor for error code
    cudaError_t code() const noexcept { return errorCode; }

    // Accessor for error string
    const std::string& str() const noexcept { return errorString; }

    // Accessor for file name
    const std::string& file() const noexcept { return fileName; }

    // Accessor for line number
    int line() const noexcept { return lineNumber; }

private:
    // Helper to build a detailed error message
    static std::string buildMessage(cudaError_t errCode,
                                    const char* errStr,
                                    const char* file,
                                    int line) {
        std::ostringstream oss;
        oss << "CUDA Error [" << static_cast<int>(errCode) << "]: "
            << errStr << " at "
            << file << ":" << line;
        return oss.str();
    }

    const cudaError_t errorCode;
    const std::string errorString;
    const std::string fileName;
    const int lineNumber;
};

// Example usage helper macro (optional)
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            throw CudaException(err, cudaGetErrorString(err),        \
                               __FILE__, __LINE__);                  \
        }                                                             \
    } while (0)

// Example kernel (no exception handling on device)
__global__ void dummyKernel() {
    // Dummy kernel
}

int main() {
    // Allocate some memory to trigger a CUDA call
    float* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, 1024 * sizeof(float)));

    // Launch dummy kernel
    dummyKernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());

    // Clean up
    CUDA_CHECK(cudaFree(d_ptr));

    return 0;
}
```