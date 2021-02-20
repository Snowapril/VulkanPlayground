// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <vk_types.h>
#include <vk_mesh.h>
#include <functional>
#include <deque>
#include <glm/glm.hpp>

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function)
	{
		deletors.push_back(function);
	}

	void flush()
	{
		for (auto it = deletors.rbegin(); it != deletors.rend(); ++it)
		{
			(*it)();
		}

		deletors.clear();
	}
};

struct MeshPushConstants
{
	glm::vec4 data;
	glm::mat4 render_matrix;
};

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

	VkRenderPass _renderPass;
	std::vector<VkFramebuffer> _framebuffers;

	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkPipelineLayout _trianglePipelineLayout;
	VkPipeline _trianglePipeline;
	VkPipeline _coloredTrianglePipeline;

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	VkPipeline _meshPipeline;
	Mesh _triangleMesh;
	Mesh _monkeyMesh;

	VkPipelineLayout _meshPipelineLayout;

	VkImageView _depthImageView;
	AllocatedImage _depthImage;
	VkFormat _depthFormat;

	int _selectedShader { 0 };

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

	bool load_shader_module(const char* filepath, VkShaderModule* outShaderModule);
private:
	void init_vulkan();

	void init_swapchain();

	void init_commands();

	void init_default_renderpass();

	void init_framebuffers();

	void init_sync_structures();

	void init_pipelines();

	void load_meshes();

	void upload_mesh(Mesh& mesh);
};

class PipelineBuilder
{
public:
	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssemblyInfo;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizationInfo;
	VkPipelineColorBlendAttachmentState _colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo _multisamplingInfo;
	VkPipelineLayout _pipelineLayout;
	VkPipelineDepthStencilStateCreateInfo _depthStencil;

	VkPipeline build_pipeline(VkDevice device, VkRenderPass renderPass);
private:
};