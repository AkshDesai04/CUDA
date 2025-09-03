/*
Instantiate your class in a scope (e.g., inside a function or an if block) and verify that the destructor is called automatically when the object goes out of scope.

Thinking:
The goal is to demonstrate that a class destructor is invoked automatically when an object goes out of scope. In a CUDA .cu file we can simply use host-side code, because the destructor behavior on the host side is the same as standard C++. We will:
1. Define a simple class `MyClass` with a constructor and destructor that print messages to indicate when they are called.
2. In `main`, create an instance of `MyClass` inside an `if` block. The destructor should run when the block ends.
3. Also call a helper function `testScope` where an instance is created locally; the destructor should run when the function returns.
4. The program prints a clear sequence of messages to confirm the automatic destruction.

This code is straightforward CUDA C++ and can be compiled with `nvcc`. It does not use any device-specific features, focusing on host-side automatic object lifetime management.
*/

#include <iostream>

// Simple class with constructor and destructor that print messages
class MyClass {
public:
    MyClass() {
        std::cout << "MyClass constructed\n";
    }
    ~MyClass() {
        std::cout << "MyClass destroyed\n";
    }
};

// Function that creates an instance of MyClass inside its own scope
void testScope() {
    std::cout << "Entering testScope\n";
    MyClass obj; // Constructor called here
    std::cout << "Inside testScope\n";
    // When function exits, destructor will be called automatically
}

int main() {
    std::cout << "Program start\n";

    // Instantiate in an if block
    if (true) {
        MyClass obj2; // Constructor called here
        std::cout << "Inside if block\n";
    } // Destructor called automatically when block ends

    // Call function that also creates an instance
    testScope();

    std::cout << "Program end\n";
    return 0;
}
