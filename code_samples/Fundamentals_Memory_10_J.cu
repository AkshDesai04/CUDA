```cpp
/*
 * Aim of the program:
 * Create a C++ class that encapsulates a 2D pitched device allocation, storing the pointer and pitch, and providing a safe access method.
 *
 * Entire thinking:
 * 
 * 1. Pitched memory is allocated with cudaMallocPitch, which returns a pointer and a pitch (row width in bytes). 2D kernels typically use this to avoid memory bank conflicts and to efficiently read/write rows.
 * 2. The class should manage lifetime (allocate in constructor, free in destructor) and expose a convenient, type-safe access method for device kernels.
 * 3. To keep the interface generic, the class will be templated on the element type T. The constructor will take width, height and use sizeof(T) internally.
 * 4. We need helper functions for error checking: a simple macro that aborts on any CUDA error, to keep the code concise.
 * 5. The access method will be an inline __device__ function that returns a reference to the element at (x, y).  It computes the row start using the pitch and the element offset within the row.  Since we return a reference, the function must be marked __device__ only.
 * 6. The class will provide host-only accessors for the raw pointer, pitch, width and height for debugging or copy operations.
 * 7. We also add a simple copyFromHost / copyToHost method for completeness, but the core requirement is the allocation and access.
 * 8. No external dependencies beyond <cuda_runtime.h> and standard headers.  The code will compile as a .cu file.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

/* Utility macro for CUDA error checking */
inline void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}

/* Templated class encapsulating 2D pitched allocation */
template <typename T>
class Pitch2DAlloc
{
private:
    T*       d_ptr;   // device pointer
    size_t   pitch;   // pitch in bytes
    int      width;   // number of elements per row
    int      height;  // number of rows

public:
    /* Constructor: allocate pitched memory */
    Pitch2DAlloc(int w, int h) : d_ptr(nullptr), pitch(0), width(w), height(h)
    {
        if (w <= 0 || h <= 0)
        {
            throw std::invalid_argument("Width and height must be positive");
        }

        cudaError_t err = cudaMallocPitch((void**)&d_ptr, &pitch, width * sizeof(T), height);
        checkCudaError(err, "cudaMallocPitch failed");
    }

    /* Destructor: free allocated memory */
    ~Pitch2DAlloc()
    {
        if (d_ptr)
        {
            cudaFree(d_ptr);
        }
    }

    /* Delete copy semantics */
    Pitch2DAlloc(const Pitch2DAlloc&) = delete;
    Pitch2DAlloc& operator=(const Pitch2DAlloc&) = delete;

    /* Enable move semantics */
    Pitch2DAlloc(Pitch2DAlloc&& other) noexcept
        : d_ptr(other.d_ptr), pitch(other.pitch), width(other.width), height(other.height)
    {
        other.d_ptr = nullptr;
        other.pitch = 0;
        other.width = 0;
        other.height = 0;
    }

    Pitch2DAlloc& operator=(Pitch2DAlloc&& other) noexcept
    {
        if (this != &other)
        {
            if (d_ptr) cudaFree(d_ptr);
            d_ptr = other.d_ptr;
            pitch = other.pitch;
            width = other.width;
            height = other.height;
            other.d_ptr = nullptr;
            other.pitch = 0;
            other.width = 0;
            other.height = 0;
        }
        return *this;
    }

    /* Accessor for raw pointer */
    T* raw() const { return d_ptr; }

    /* Accessor for pitch in bytes */
    size_t getPitch() const { return pitch; }

    /* Accessor for dimensions */
    int getWidth()  const { return width;  }
    int getHeight() const { return height; }

    /* Safe device access: returns reference to element at (x, y) */
    __device__ T& at(int x, int y)
    {
        // Bounds checking is omitted for performance; user should ensure (x, y) are valid
        char* row = (char*)d_ptr + y * pitch;
        return *(reinterpret_cast<T*>(row + x * sizeof(T)));
    }

    /* Optional: copy data from host to device */
    void copyFromHost(const T* host_ptr, bool rowMajor = true)
    {
        cudaError_t err;
        if (rowMajor)
        {
            err = cudaMemcpy2D(d_ptr, pitch, host_ptr, width * sizeof(T),
                               width * sizeof(T), height, cudaMemcpyHostToDevice);
        }
        else
        {
            // column-major copy: use transpose logic (not implemented here for brevity)
            throw std::runtime_error("Column-major copy not implemented");
        }
        checkCudaError(err, "cudaMemcpy2D (copyFromHost) failed");
    }

    /* Optional: copy data from device to host */
    void copyToHost(T* host_ptr, bool rowMajor = true) const
    {
        cudaError_t err;
        if (rowMajor)
        {
            err = cudaMemcpy2D(host_ptr, width * sizeof(T), d_ptr, pitch,
                               width * sizeof(T), height, cudaMemcpyDeviceToHost);
        }
        else
        {
            throw std::runtime_error("Column-major copy not implemented");
        }
        checkCudaError(err, "cudaMemcpy2D (copyToHost) failed");
    }
};

/* Example kernel that sets all elements to a given value */
__global__ void initKernel(Pitch2DAlloc<float> mat, float val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < mat.getWidth() && y < mat.getHeight())
    {
        mat.at(x, y) = val;
    }
}

/* Simple test harness */
int main()
{
    const int WIDTH = 1024;
    const int HEIGHT = 512;

    Pitch2DAlloc<float> mat(WIDTH, HEIGHT);

    // Launch kernel to initialize matrix
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initKernel<<<numBlocks, threadsPerBlock>>>(mat, 3.14f);
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");

    // Optional: copy back to host and verify
    float* hostData = new float[WIDTH * HEIGHT];
    mat.copyToHost(hostData);
    if (hostData[0] == 3.14f)
    {
        printf("Success: first element is %.2f\n", hostData[0]);
    }
    else
    {
        printf("Failure: first element is %.2f\n", hostData[0]);
    }

    delete[] hostData;
    return 0;
}
```