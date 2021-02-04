#include <iostream>
#include <cstdlib>
#include "VulkanApp.hpp"

int main(int argc, char* argv[])
{
    VulkanApp app;
    (void)argc;
    (void)argv;

    try {
        app.Run(); 
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}