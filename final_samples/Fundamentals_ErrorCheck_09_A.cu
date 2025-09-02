```cpp
/*
Aim of the program:
Define a custom exception class, e.g., `CudaException`, that inherits from `std::runtime_error` and stores the error code, error string, file, and line.

Thinking and design:
- The exception class should provide detailed context for CUDA errors.
- It will store the CUDA error code (`cudaError_t`), the human‑readable error string (`std::string`), and the source file and line where the error was thrown.
- The constructor will take the error code, file, and line, and it will automatically look up the CUDA error string via `cudaGetErrorString`.
- The exception message (`what()`) will combine all information in a single string: `[file:line] CUDA error code X: error string`.
- Provide accessor methods to retrieve the stored error code, error string, file, and line for programmatic inspection.
- Inherit from `std::runtime_error` so that it can be used with standard exception handling mechanisms.
- Include necessary headers: `<stdexcept>`, `<string>`, `<sstream>`, and `<cuda_runtime.h>`.
- The code will be a self‑contained .cu file suitable for inclusion in a larger CUDA project.

Below is the complete implementation of `CudaException` in a `.cu` file.
*/

#include <stdexcept>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

class CudaException : public std::runtime_error {
public:
    // Constructor that accepts a CUDA error code, file, and line.
    CudaException(cudaError_t errCode, const char* file, int line)
        : std::runtime_error(buildMessage(errCode, file, line)),
          errorCode_(errCode),
          errorString_(cudaGetErrorString(errCode)),
          file_(file),
          line_(line)
    {}

    // Return the CUDA error code.
    cudaError_t code() const noexcept { return errorCode_; }

    // Return the CUDA error string.
    const std::string& errorString() const noexcept { return errorString_; }

    // Return the file name where the exception was thrown.
    const std::string& file() const noexcept { return file_; }

    // Return the line number where the exception was thrown.
    int line() const noexcept { return line_; }

private:
    // Helper to build the exception message string.
    static std::string buildMessage(cudaError_t errCode, const char* file, int line) {
        std::ostringstream oss;
        oss << "[" << file << ":" << line << "] "
            << "CUDA error code " << static_cast<int>(errCode)
            << ": " << cudaGetErrorString(errCode);
        return oss.str();
    }

    cudaError_t errorCode_;
    std::string errorString_;
    std::string file_;
    int line_;
};
```