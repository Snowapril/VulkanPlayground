#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class VulkanApp
{
public:
    void Run();

private:
    void InitWindow();
    void InitVulkan();
    void MainLoop();
    void CleanUp();

    GLFWwindow* _window;
};