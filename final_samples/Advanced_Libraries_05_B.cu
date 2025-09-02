```cpp
/*
Aim of the program:
Use CUB's `DeviceRadixSort::SortPairs` to sort key-value pairs.

Thought process:
1. We need to demonstrate sorting key-value pairs on the GPU using the CUB library's
   DeviceRadixSort::SortPairs routine.  
2. CUB's radix sort requires two passes: first, a query pass with a null temporary
   storage pointer to determine the amount of temporary storage needed, and second,
   an actual sort pass with the allocated storage.  
3. We'll create a small example on the host: an array of integer keys and
   corresponding integer values.  
4. These arrays are copied to device memory.  
5. We'll then invoke DeviceRadixSort::SortPairs twice: once to obtain the required
   storage size, allocate it, and once to perform the actual sorting.  
6. Finally, we copy the sorted keys and values back to the host and print them
   to verify that the sort worked correctly.  
7. For simplicity and portability, we use CUDA's basic error checking macro to
   catch any runtime errors.  
8. The code is fully self‑contained and can be compiled with nvcc:
   nvcc -arch=sm_70 -O2 radix_sort_pairs.cu -o radix_sort_pairs
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cub/cub.cuh>

// Simple CUDA error checking macro
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in file '" << __FILE__       \
                      << "' in line " << __LINE__ << ": "          \
                      << cudaGetErrorString(err) << std::endl;    \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main()
{
    // Host data: keys and values
    const int num_items = 10;
    std::vector<int> h_keys  = { 12,  5,  9,  7,  3, 15, 1,  8, 13, 4 };
    std::vector<int> h_vals  = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };

    // Device pointers
    int *d_keys = nullptr;
    int *d_vals = nullptr;
    int *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate device memory for keys and values
    CHECK_CUDA( cudaMalloc((void**)&d_keys,  num_items * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**)&d_vals,  num_items * sizeof(int)) );

    // Copy data from host to device
    CHECK_CUDA( cudaMemcpy(d_keys, h_keys.data(),  num_items * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_vals, h_vals.data(),  num_items * sizeof(int), cudaMemcpyHostToDevice) );

    // ---------- Query pass: get required temporary storage size ----------
    cub::DeviceRadixSort::SortPairs<int, int>(nullptr, temp_storage_bytes,
                                              d_keys, d_keys,   // in/out
                                              d_vals, d_vals,   // in/out
                                              num_items);

    // Allocate temporary storage
    CHECK_CUDA( cudaMalloc(&d_temp_storage, temp_storage_bytes) );

    // ---------- Sort pass ----------
    cub::DeviceRadixSort::SortPairs<int, int>(d_temp_storage, temp_storage_bytes,
                                              d_keys, d_keys,   // in/out
                                              d_vals, d_vals,   // in/out
                                              num_items);

    // Copy sorted results back to host
    std::vector<int> sorted_keys(num_items);
    std::vector<int> sorted_vals(num_items);
    CHECK_CUDA( cudaMemcpy(sorted_keys.data(), d_keys, num_items * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(sorted_vals.data(), d_vals, num_items * sizeof(int), cudaMemcpyDeviceToHost) );

    // Print results
    std::cout << "Sorted key-value pairs:" << std::endl;
    for (int i = 0; i < num_items; ++i)
    {
        std::cout << "Key: " << sorted_keys[i] << "  Value: " << sorted_vals[i] << std::endl;
    }

    // Cleanup
    CHECK_CUDA( cudaFree(d_keys) );
    CHECK_CUDA( cudaFree(d_vals) );
    CHECK_CUDA( cudaFree(d_temp_storage) );

    return 0;
}
```