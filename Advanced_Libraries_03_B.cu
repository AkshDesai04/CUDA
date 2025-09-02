/*
Use thrust::min_element and thrust::max_element to find the minimum and maximum values in a device_vector

Thought Process:
1. The goal is to demonstrate the use of Thrust's `min_element` and `max_element` algorithms on a `thrust::device_vector`.  
2. We need a complete, self‑contained CUDA program that compiles with `nvcc`.  
3. Include the necessary Thrust headers: `<thrust/host_vector.h>`, `<thrust/device_vector.h>`, and `<thrust/extrema.h>` (where the min/max algorithms live).  
4. Create a `thrust::host_vector` with some sample data, then copy it to a `thrust::device_vector`.  
5. Call `thrust::min_element` and `thrust::max_element` on the device vector.  These return iterators pointing to the min/max elements.  
6. Dereference the iterators to obtain the actual values and print them.  
7. Use `std::cout` for output.  
8. Wrap everything in `main()` and include a simple error check for CUDA runtime (optional but good practice).  
9. Ensure the program compiles as a `.cu` file and can be run with `nvcc -o minmax minmax.cu && ./minmax`.

*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <iostream>
#include <vector>

int main() {
    // Sample data on host
    std::vector<int> host_vec = { 12, -3, 45, 7, 0, -18, 23, 9, 14, -2 };
    
    // Transfer to device
    thrust::device_vector<int> dev_vec(host_vec.begin(), host_vec.end());

    // Find minimum element
    thrust::device_vector<int>::iterator min_it = thrust::min_element(dev_vec.begin(), dev_vec.end());
    // Find maximum element
    thrust::device_vector<int>::iterator max_it = thrust::max_element(dev_vec.begin(), dev_vec.end());

    // Dereference iterators to get values
    int min_val = *min_it;
    int max_val = *max_it;

    std::cout << "Minimum value in device_vector: " << min_val << std::endl;
    std::cout << "Maximum value in device_vector: " << max_val << std::endl;

    return 0;
}
