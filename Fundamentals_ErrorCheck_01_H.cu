```cpp
/*
Aim of the program:
In C++, change the macro to throw a custom exception instead of exiting.

Thinking process:
- The task is to replace a macro that previously called exit() with one that throws a custom exception.
- Since the file is a .cu file (CUDA C++), we can still use standard C++ exception handling in host code.
- We'll create a simple custom exception class `CudaCheckException` that derives from std::exception.
- The macro `CUDA_CHECK` will evaluate an expression; if it is false, it will throw `CudaCheckException` with a message containing the expression, file, and line.
- The macro will be safe for both host and device code where possible; however, exceptions cannot be thrown from device code, so this macro should only be used in host functions. We'll document that.
- We'll also provide a simple host function that demonstrates using the macro and catching the exception.
- The kernel code will remain trivial; the focus is on the macro and exception handling on the host side.
- We'll keep the file self-contained and compilable with `nvcc`.

*/

// Custom exception class for CUDA check failures
class CudaCheckException : public std::exception {
public:
    explicit CudaCheckException(const char* msg) : message_(msg) {}
    explicit CudaCheckException(const std::string& msg) : message_(msg) {}
    virtual const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};

// Macro to check a condition on the host side.
// If the condition fails, it throws CudaCheckException.
// Note: Throwing exceptions from device code is not supported.
#define CUDA_CHECK(expr) \
    do { \
        if (!(expr)) { \
            throw CudaCheckException( \
                std::string("CUDA_CHECK failed: ") + #expr + \
                " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

// Sample kernel that does nothing significant
__global__ void dummyKernel() {
    // Just a placeholder kernel
}

// Host function that uses the CUDA_CHECK macro
void hostFunction(bool condition) {
    // This function intentionally uses the macro to demonstrate exception handling
    CUDA_CHECK(condition);  // If condition is false, exception is thrown
    // If the condition is true, proceed
    printf("Condition passed. Continuing hostFunction.\n");
}

int main(int argc, char* argv[]) {
    // Launch dummy kernel for demonstration
    dummyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Demonstrate CUDA_CHECK with a passing condition
    try {
        hostFunction(true);
    } catch (const CudaCheckException& e) {
        fprintf(stderr, "Caught exception: %s\n", e.what());
    }

    // Demonstrate CUDA_CHECK with a failing condition
    try {
        hostFunction(false);  // This will throw
    } catch (const CudaCheckException& e) {
        fprintf(stderr, "Caught exception: %s\n", e.what());
    }

    return 0;
}
```