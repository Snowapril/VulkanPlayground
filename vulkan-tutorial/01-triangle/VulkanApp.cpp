#include "VulkanApp.hpp"
#include <glm/glm.hpp>
#include <iostream>
#include <thread>
#include <exception>
#include <chrono>
#include <cstring>

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

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    std::cout << "Available Extensions" << std::endl;
    for (const auto& extension : extensions)
    {
        std::cout << '\t' << extension.extensionName << std::endl;
    }

    #ifdef NDEBUG
        _enableValidationLayers = true;
    #else
        _enableValidationLayers = false;
    #endif

    CreateInstance();
    SetDebugMessenger();
    PickPhysicalDevice();
}

bool VulkanApp::CheckValidationLayerSupport()
{
    //! Check Validation Layers Support
    unsigned int layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : _validationLayers)
    {
        bool layerFound = false;
        for (const auto& layerProperty : availableLayers)
        {
            if (strcmp(layerName, layerProperty.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }
        if (!layerFound)
        {
            return false;
        }
    }
    return true;
}

std::vector<const char*> VulkanApp::GetRequiredExtensions()
{
    unsigned int glfwExtensionCount;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (_enableValidationLayers)
    {
        //! VK_EXT_DEBUG_UTILS_EXTENSION_NAME == VK_EXT_debug_utils
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return extensions;
}

void VulkanApp::CreateInstance()
{
    if (_enableValidationLayers && !CheckValidationLayerSupport())
    {
        throw std::runtime_error("Validation Layers Requested, But not Avaliable");
    }
    
    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = GetRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<unsigned int>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    if (_enableValidationLayers)
    {
        createInfo.enabledLayerCount = _validationLayers.size();
        createInfo.ppEnabledLayerNames = _validationLayers.data();
        PopulateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else
    {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &_instance) != VK_SUCCESS) 
        throw std::runtime_error("Failed to create vulkan instance");
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanApp::DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) 
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

void VulkanApp::PickPhysicalDevice()
{
    unsigned int deviceCount { 0 };
    vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);

    if (deviceCount == 0) 
    {
        throw std::runtime_error("Cannot find GPUs with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());

    for (const auto& device : devices)
    {
        if (IsDeviceSuitable(device))
        {
            _physicalDevice = device;
            break;
        }
    }

    if (_physicalDevice == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Failed to find suitable GPU");
    }
}

bool VulkanApp::IsDeviceSuitable(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
    //! dedicated graphics card and support for geometry shader.
    return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader;
}

VkResult VulkanApp::CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void VulkanApp::DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

void VulkanApp::PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = {}; //! Without this initialization, dummy variable in createInfo struct will crash program.
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugCallback;
    createInfo.pUserData = nullptr;
}

void VulkanApp::SetDebugMessenger()
{
    if (!_enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo {};
    PopulateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(_instance, &createInfo, nullptr, &_debugMessenger) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to load debug messenger");
    }
}

void VulkanApp::MainLoop()
{
    int counter = 3;
    while (glfwWindowShouldClose(_window) == GLFW_FALSE && counter--) 
    {
        glfwPollEvents();
        std::cout << "MainLoop" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void VulkanApp::CleanUp()
{
    if (_debugMessenger) DestroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);
    if (_instance) vkDestroyInstance(_instance, nullptr);
    glfwDestroyWindow(_window);
    glfwTerminate();
}
