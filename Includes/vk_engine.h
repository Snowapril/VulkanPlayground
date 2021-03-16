// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <unordered_map>
#include <vector>
#include <vk_types.h>
#include <vk_mesh.h>
#include <functional>
#include <deque>
#include <glm/glm.hpp>

//! Note that we store the VkPipeline and layout by value, not by pointer
//! They are 64 bit handles to internal driver structures anyway so storing pointer to them isn't very useful
struct Material
{
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSet textureSet { VK_NULL_HANDLE };
};

struct RenderObject
{
	Mesh* mesh;
	Material* material;
	glm::mat4 modelMatrix;
};

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

struct GPUCameraData
{
	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 viewProj;
};

struct GPUSceneData
{
	glm::vec4 fogColor; //! w is for exponent
	glm::vec4 fogDistances; //! x for min, y for max, zw unused.
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; //! w for power
	glm::vec4 sunlightColor;
}; //! 80 bytes with alignment.

struct GPUObjectData
{
	glm::mat4 modelMatrix;
};

struct Texture
{
	AllocatedImage image;
	VkImageView imageView;
};

struct UploadContext
{
	VkFence _uploadFence;
	VkCommandPool _commandPool;
};

struct FrameData
{
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	//! Buffer that holds a single GPUCameraData to use when rendering
	AllocatedBuffer cameraBuffer;
	AllocatedBuffer objectBuffer;

	VkDescriptorSet globalDescriptor;
	VkDescriptorSet objectDescriptor;
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
	VkPhysicalDeviceProperties _gpuProperties;
	VkDevice _device; // vulkan device for commands
	VkSurfaceKHR _surface; // vulkan window surface

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat; // image format expected by the window system.
	std::vector<VkImage> _swapchainImages; // array of images from the swapchain
	std::vector<VkImageView> _swapchainImageViews; // array of image views from the swap chain

	VkQueue _graphicsQueue; // queue we will submit to
	unsigned int _graphicsQueueFamily; //family of that queue

	VkRenderPass _renderPass;
	std::vector<VkFramebuffer> _framebuffers;

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

	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;
	VkDescriptorSetLayout _singleTextureLayout;
	VkDescriptorPool _descriptorPool;

	std::vector<RenderObject> _renderables;
	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;

	UploadContext _uploadContext;

	GPUSceneData _sceneParameters;
	AllocatedBuffer _sceneParameterBuffer;

	static constexpr unsigned int kFrameOverlap = 2;
	//! frame storage
	FrameData _frames[kFrameOverlap];

	//! getter for the frame we are rendering to right now
	FrameData& get_current_frame();

	int _selectedShader { 0 };

	std::unordered_map<std::string, Texture> _loadedTextures;

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

	void immeidate_submit(std::function<void(VkCommandBuffer)>&& function);

	bool load_shader_module(const char* filepath, VkShaderModule* outShaderModule);
	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags flags, VmaMemoryUsage usage);
private:
	void init_vulkan();

	void init_swapchain();

	void init_commands();

	void init_default_renderpass();

	void init_framebuffers();

	void init_sync_structures();

	void init_pipelines();

	void load_images();

	void load_meshes();

	void init_scene();

	void upload_mesh(Mesh& mesh);

	size_t pad_uniform_buffer_size(size_t originalSize);


	void init_descriptors();

	Material* create_material(VkPipeline pipeline, VkPipelineLayout pipelineLayout, const std::string& name);
	Material* get_material(const std::string& name);
	Mesh* get_mesh(const std::string& name);

	void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);
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