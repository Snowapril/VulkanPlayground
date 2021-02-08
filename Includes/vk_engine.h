// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <vk_types.h>

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber {0};

	VkExtent2D _windowExtent{ 1200 , 900 };

	struct SDL_Window* _window{ nullptr };
	
	VkInstance _instance; //vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger; // vulkan debug output handle
	VkPhysicalDevice _chosenGPU; // GPU chosen as the default device
	VkDevice _device; // vulkan device for commands
	VkSurfaceKHR _surface; // vulkan window surface

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat; // image format expected by the window system.
	std::vector<VkImage> _swapchainImages; // array of images from the swapchain
	std::vector<VkImageView> _swapchainImageViews; // array of image views from the swap chain

	VkQueue _graphicsQueue; // queue we will submit to
	unsigned int _graphicsQueueFamily; //family of that queue

	VkCommandPool _commandPool; //the common pool for our commands
	VkCommandBuffer _commandBuffer; // the buffer we will record into


	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();
private:
	void init_vulkan();

	void init_swapchain();

	void init_commands();
};