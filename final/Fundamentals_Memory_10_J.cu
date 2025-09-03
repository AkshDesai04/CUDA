/*
Aim: Create a C++ class that encapsulates a 2D pitched device allocation, storing the pointer and pitch, and providing a safe access method.

Thinking:  
The goal is to provide a convenient, RAII‑style wrapper around CUDA’s 2‑D pitched allocation (`cudaMallocPitch`) so that the user can easily allocate, free, and access a 2‑D array on the device.  
The wrapper should:
1. Store the device pointer (`void*`) and the pitch in bytes.
2. Keep track of the logical width and height (in elements, not bytes).
3. Allocate memory in the constructor and free it in the destructor.
4. Delete the copy constructor/assignment to avoid accidental double free, but provide move semantics for efficient transfers.
5. Expose a `__device__` method `at(x, y)` that returns a pointer to the element at the requested coordinates using the stored pitch.
6. Provide convenient host helpers to copy data to/from the device (`copyFromHost`, `copyToHost`).
7. Provide accessors for the raw device pointer, pitch, width, and height.  
   Those that are needed on the device side (`at`, `getWidth`, `getHeight`) are marked `__device__`.  
8. Add a simple test in `main` that allocates a 10×5 array, copies zeroes in, launches a kernel that sets each element to 42, copies back, and prints the result.

The resulting code is a self‑contained .cu file that can be compiled with `nvcc`. 
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

/*--------------------------- PitchedArray2D class ---------------------------*/
template <typename T>
class PitchedArray2D
{
public:
    // Constructor: allocate pitched memory
    PitchedArray2D(size_t w, size_t h)
        : d_ptr(nullptr), pitch(0), width(w), height(h)
    {
        size_t bytesPerRow = w * sizeof(T);
        cudaError_t err = cudaMallocPitch(&d_ptr, &pitch, bytesPerRow, h);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMallocPitch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Destructor: free memory
    ~PitchedArray2D()
    {
        if (d_ptr)
            cudaFree(d_ptr);
    }

    // Delete copy semantics
    PitchedArray2D(const PitchedArray2D&) = delete;
    PitchedArray2D& operator=(const PitchedArray2D&) = delete;

    // Move semantics
    PitchedArray2D(PitchedArray2D&& other) noexcept
        : d_ptr(other.d_ptr), pitch(other.pitch),
          width(other.width), height(other.height)
    {
        other.d_ptr = nullptr;
        other.pitch = 0;
    }

    PitchedArray2D& operator=(PitchedArray2D&& other) noexcept
    {
        if (this != &other)
        {
            if (d_ptr)
                cudaFree(d_ptr);
            d_ptr   = other.d_ptr;
            pitch   = other.pitch;
            width   = other.width;
            height  = other.height;
            other.d_ptr = nullptr;
            other.pitch = 0;
        }
        return *this;
    }

    /*--------------------------- Device side access ---------------------------*/
    __device__ T* at(size_t x, size_t y)
    {
        // Bounds are the caller’s responsibility; no runtime check here for speed
        return reinterpret_cast<T*>((char*)d_ptr + y * pitch) + x;
    }

    __device__ size_t getWidth() const { return width; }
    __device__ size_t getHeight() const { return height; }

    /*--------------------------- Host side helpers ---------------------------*/
    // Copy from host to device
    void copyFromHost(const T* hostPtr, size_t w, size_t h)
    {
        if (w > width || h > height)
            throw std::runtime_error("copyFromHost: source size exceeds allocation");
        cudaMemcpy2D(d_ptr, pitch, hostPtr, w * sizeof(T),
                     w * sizeof(T), h, cudaMemcpyHostToDevice);
    }

    // Copy from device to host
    void copyToHost(T* hostPtr, size_t w, size_t h) const
    {
        if (w > width || h > height)
            throw std::runtime_error("copyToHost: destination size exceeds allocation");
        cudaMemcpy2D(hostPtr, w * sizeof(T), d_ptr, pitch,
                     w * sizeof(T), h, cudaMemcpyDeviceToHost);
    }

    // Raw device pointer (used for kernels that cannot accept class objects)
    T* devicePtr() const { return static_cast<T*>(d_ptr); }
    size_t pitchBytes() const { return pitch; }
    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }

private:
    void*   d_ptr;   // Device pointer to the pitched array
    size_t  pitch;   // Pitch in bytes
    size_t  width;   // Logical width in elements
    size_t  height;  // Logical height in elements
};

/*--------------------------- Example kernel ---------------------------*/
__global__ void setValue(PitchedArray2D<int> arr)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < arr.getWidth() && y < arr.getHeight())
    {
        *(arr.at(x, y)) = 42;  // Set each element to 42
    }
}

/*--------------------------- Main function ---------------------------*/
int main()
{
    const size_t width  = 10;
    const size_t height = 5;

    try
    {
        // Allocate device array
        PitchedArray2D<int> d_arr(width, height);

        // Initialize host data
        std::vector<int> h_input(width * height, 0);
        d_arr.copyFromHost(h_input.data(), width, height);

        // Launch kernel
        dim3 threads(16, 16);
        dim3 blocks((width + threads.x - 1) / threads.x,
                    (height + threads.y - 1) / threads.y);
        setValue<<<blocks, threads>>>(d_arr);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            return EXIT_FAILURE;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
            return EXIT_FAILURE;
        }

        // Copy result back to host
        std::vector<int> h_output(width * height);
        d_arr.copyToHost(h_output.data(), width, height);

        // Print result
        std::cout << "Resulting 2D array:" << std::endl;
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                std::cout << h_output[y * width + x] << '\t';
            }
            std::cout << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
