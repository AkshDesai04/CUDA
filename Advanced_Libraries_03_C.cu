```cuda
/*
Use a `thrust::counting_iterator` to initialize a `device_vector` with the sequence 0, 1, 2, ... without using any host data.

Thinking:
- The goal is to show how to fill a thrust::device_vector directly on the GPU without transferring data from the host.
- thrust::counting_iterator provides a lazy range of integers starting from a given value. It can be used as a source range in algorithms like thrust::copy.
- By creating a device_vector of desired size, we can copy the sequence from the counting iterator into it.
- No host array or vector is needed for initialization; the device_vector is populated entirely on the device.
- After initialization, we can optionally copy the data back to a host_vector to verify the contents or print some elements.
- The program demonstrates basic thrust usage and a common pattern for initializing device memory efficiently.
*/

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <iostream>

int main(void)
{
    const int N = 10;  // size of the sequence

    // Allocate device vector of size N
    thrust::device_vector<int> d_vec(N);

    // Use counting_iterator as source range [0, N)
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(N);

    // Copy sequence directly into device vector
    thrust::copy(begin, end, d_vec.begin());

    // Copy back to host for verification (optional)
    thrust::host_vector<int> h_vec = d_vec;

    // Print the result
    std::cout << "Device vector initialized with counting_iterator:\n";
    for (int i = 0; i < N; ++i)
    {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```