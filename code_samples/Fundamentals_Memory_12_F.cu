```cpp
/*
Aim of the program: 
Instantiate your class in a scope (e.g., inside a function or an `if` block) and verify that the destructor is called automatically when the object goes out of scope.

Thinking:
- Define a simple C++ class with constructor and destructor that print messages to std::cout.
- Instantiate the class inside an `if` block in main to create a local scope.
- The destructor should be invoked automatically when the block ends.
- The program is compiled as a CUDA source (.cu) but all operations are host-side; CUDA-specific code is not needed for this demonstration.
- We keep the code minimal and portable, using only <iostream> and standard C++ features.
*/

#include <iostream>

class MyClass {
public:
    MyClass() {
        std::cout << "Constructor called\n";
    }
    ~MyClass() {
        std::cout << "Destructor called\n";
    }
};

int main() {
    std::cout << "Entering if block\n";
    if (true) {
        MyClass obj; // Constructor is called here
        std::cout << "Inside if block\n";
    } // Destructor is called automatically here when obj goes out of scope
    std::cout << "After if block\n";
    return 0;
}
```