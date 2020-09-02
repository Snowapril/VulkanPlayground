#include "VulkanApp.hpp"
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <iostream>
#include <thread>
#include <chrono>

//! https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance

void VulkanApp::Run()
{
    InitVulkan();
    MainLoop();
    CleanUp();
}

void VulkanApp::InitVulkan()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize glfw" << std::endl;
        return ;
    }        

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); //! Make glfw not to create opengl context.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    _window = glfwCreateWindow(800, 600, "01-triangle", nullptr, nullptr);

    unsigned int extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::cout << "#Vulkan Extension Count : " << extensionCount << std::endl;
}

void VulkanApp::MainLoop()
{
    while (glfwWindowShouldClose(_window) == false)
    {
        glfwPollEvents();
        std::cout << "MainLoop" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void VulkanApp::CleanUp()
{
    glfwDestroyWindow(_window);
    glfwTerminate();
}