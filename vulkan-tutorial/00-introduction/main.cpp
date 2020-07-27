#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>

int main(int argc, char* argv[])
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize glfw" << std::endl;
        return 1;
    }        

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "00-introduction", nullptr, nullptr);

    unsigned int extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::cout << "#Vulkan Extension Count : " << extensionCount << std::endl;

    int count = 10;
    while (--count)
    {
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}