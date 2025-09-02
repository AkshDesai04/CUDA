/*
Aim: Define a custom comparison operator and pass it to `thrust::sort` to sort in descending order.

Thinking:
1. I need a CUDA source file (.cu) that demonstrates how to sort data in descending order using Thrust.
2. The goal is to show how to provide a custom comparator to `thrust::sort`.  
3. I will use `thrust::device_vector` to store integer data on the GPU.
4. The custom comparator can be a simple functor with an `operator()` that returns true when the first argument is greater than the second – this makes the sort descending.
5. I will create a small set of sample data, copy it to the device, perform the sort with the comparator, then copy the result back to the host for printing.
6. Printing is done by copying the device vector back into a `thrust::host_vector` or a standard `std::vector` and outputting the elements.
7. The program will include necessary Thrust headers, the functor, and a `main` function that performs all steps. 
8. The code should compile with `nvcc` and run without any external dependencies besides the Thrust library.
*/

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

// Custom comparator functor for descending order
struct greater_than
{
    __host__ __device__
    bool operator()(const int &a, const int &b) const
    {
        return a > b; // true if a should come before b in descending order
    }
};

int main()
{
    // Sample data to sort
    thrust::host_vector<int> h_vec(10);
    h_vec[0] =  3; h_vec[1] =  7; h_vec[2] =  2; h_vec[3] =  9;
    h_vec[4] =  1; h_vec[5] =  5; h_vec[6] =  8; h_vec[7] =  4;
    h_vec[8] = 10; h_vec[9] =  6;

    std::cout << "Original vector:" << std::endl;
    for (int val : h_vec)
        std::cout << val << " ";
    std::cout << std::endl;

    // Transfer data to device
    thrust::device_vector<int> d_vec = h_vec;

    // Sort in descending order using custom comparator
    thrust::sort(d_vec.begin(), d_vec.end(), greater_than());

    // Copy sorted data back to host
    thrust::host_vector<int> h_sorted = d_vec;

    std::cout << "Sorted vector (descending):" << std::endl;
    for (int val : h_sorted)
        std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}
