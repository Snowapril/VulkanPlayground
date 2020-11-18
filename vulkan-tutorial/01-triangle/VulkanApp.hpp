#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>

class VulkanApp
{
public:
    void Run();

private:
    void InitWindow();
    void InitVulkan();
    void MainLoop();
    void CleanUp();

    void CreateInstance();
    bool CheckValidationLayerSupport();
    void SetDebugMessenger();
    void PickPhysicalDevice();
    std::vector<const char*> GetRequiredExtensions();

    struct QueueFamilyIndices {
        std::optional<unsigned int> graphicsFamily;

        bool IsComplete() const {
            return graphicsFamily.has_value();
        };
    };

    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device);

    bool IsDeviceSuitable(VkPhysicalDevice device);
    void CreateLogicalDevice();
    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
                VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                void* pUserData);
    void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
    GLFWwindow* _window;
    const std::vector<const char*> _validationLayers { "VK_LAYER_KHRONOS_validation" };
    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debugMessenger;
    VkPhysicalDevice _physicalDevice { VK_NULL_HANDLE };
    VkDevice _device;
    VkQueue _graphicsQueue;
    bool _enableValidationLayers;
};