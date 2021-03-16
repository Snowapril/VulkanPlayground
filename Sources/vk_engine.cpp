
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_texture.h>
#include <vk_initializers.h>

#include <VkBootstrap.h>

#include <iostream>
#include <fstream>

#pragma warning(push)
#pragma warning(disable : 4100)
#pragma warning(disable : 4324)
#pragma warning(disable : 4127)
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>
#pragma warning(pop)

#include <glm/gtx/transform.hpp>

#define VK_CHECK(x) \
	do \
	{ \
		VkResult err = x; \
		if (err) \
		{ \
			std::cout << "Detected Vulkan Error : " << err << std::endl; \
			std::abort(); \
		} \
	} while (0)

void VulkanEngine::init()
{
	// We initialize SDL and create a window with it. 
	SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);
	
	_window = SDL_CreateWindow(
		"Vulkan Engine",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		_windowExtent.width,
		_windowExtent.height,
		window_flags
	);
	
	//load the core vulkan structures
	init_vulkan();

	//create the swapchian
	init_swapchain();

	//initialize the commands
	init_commands();

	//initialize the default renderpass
	init_default_renderpass();

	//init framebuffers
	init_framebuffers();

	//init sync structures
	init_sync_structures();

	//init descriptors
	init_descriptors();

	//init pipelines
	init_pipelines();

	//load meshes
	load_meshes();

	load_images();

	//init scene
	init_scene();


	//everything went fine
	_isInitialized = true;
}

void VulkanEngine::init_vulkan()
{
	vkb::InstanceBuilder builder;
	//make the vulkan instance with basic debug features.
	auto inst_ret = builder.set_app_name("Example vulkan application")
		.request_validation_layers(true)
		.require_api_version(1, 1, 0)
		.use_default_debug_messenger()
		.build();

	vkb::Instance vkb_instance = inst_ret.value();

	_instance = vkb_instance.instance;
	_debug_messenger = vkb_instance.debug_messenger;

	// get the surface of the window opened with the SDL2
	SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

	// use vkbootstrap to select a GPU
	// we want a GPU that can write to the SDL surface and vulkan 1.1 supported.
	vkb::PhysicalDeviceSelector selector{ vkb_instance };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 1)
		.set_surface(_surface)
		.select()
		.value();

	// create a final vulkan device
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };
	vkb::Device vkbDevice = deviceBuilder.build().value();

	// Get the vkDevice handle used in the rest of a vulkan application
	_device = vkbDevice.device;
	_chosenGPU = physicalDevice.physical_device;

	vkGetPhysicalDeviceProperties(_chosenGPU, &_gpuProperties);
	std::cout << "The GPU has a minimum buffer alignment of " << _gpuProperties.limits.minUniformBufferOffsetAlignment << std::endl;

	// use vkbootstrap to grabbing the queue
	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	//! Initialize the memory allocator.
	VmaAllocatorCreateInfo allocator_info = {};
	allocator_info.device = _device;
	allocator_info.physicalDevice = _chosenGPU;
	allocator_info.instance = _instance;
	vmaCreateAllocator(&allocator_info, &_allocator);

	_mainDeletionQueue.push_function([=]() {
		vmaDestroyAllocator(_allocator);
	});
}

void VulkanEngine::init_swapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU, _device, _surface };
	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		//use vsync present mode
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(_windowExtent.width, _windowExtent.height)
		.build()
		.value();

	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews = vkbSwapchain.get_image_views().value();

	_swapchainImageFormat = vkbSwapchain.image_format;

	//! depth image size will match the window.
	VkExtent3D depthImageExtent = {
		_windowExtent.width,
		_windowExtent.height,
		1
	};

	//! hardcoding the depth format to 32bit float
	_depthFormat = VK_FORMAT_D32_SFLOAT;

	//! The depth image will be an image with the format we selected and depth attachment usage flags.
	VkImageCreateInfo image_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

	//! for the depth image, we want to allocate it from GPU local memory
	VmaAllocationCreateInfo dimg_allocate_info = {};
	dimg_allocate_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocate_info.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	//! allocate and create the image
	vmaCreateImage(_allocator, &image_info, &dimg_allocate_info, &_depthImage._image, &_depthImage._allocation, nullptr);

	//! build an image-view for the depth image to use for rendering
	VkImageViewCreateInfo image_view_info = vkinit::image_view_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

	VK_CHECK(vkCreateImageView(_device, &image_view_info, nullptr, &_depthImageView));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, _depthImageView, nullptr);
		vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
		vkDestroySwapchainKHR(_device, _swapchain, nullptr);
	});
}

void VulkanEngine::init_commands()
{
	//! create a command pool for commands submitted into the graphics queue
	//! we also want the pool to allow for resetting of individual command buffers
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	for (auto& frame : _frames)
	{
		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &frame._commandPool));

		//! allocate the default buffer that will be used for rendering
		VkCommandBufferAllocateInfo commandBufferInfo = vkinit::command_buffer_allocate_info(frame._commandPool, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		VK_CHECK(vkAllocateCommandBuffers(_device, &commandBufferInfo, &frame._mainCommandBuffer));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyCommandPool(_device, frame._commandPool, nullptr);
		});
	}

	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
	VK_CHECK(vkCreateCommandPool(_device, &uploadCommandPoolInfo, nullptr, &_uploadContext._commandPool));
	_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(_device, _uploadContext._commandPool, nullptr);
	});
}

void VulkanEngine::init_default_renderpass()
{
	//! The renderpass will use this color attachment
	VkAttachmentDescription color_attachment = {};
	//! The attachment will have the format needed by the swapchain.
	color_attachment.format = _swapchainImageFormat;
	//! 1 sample. We will not use MSAA
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	//! We clear when this attachment is loaded.
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	//! We keep the attachment stored when the renderpass ends
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	//! We dont care about the stencil
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	//! We dont know or care about the starting layout of the attachment
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	//! after the renderpass ends, the image has to be on a layout ready for display
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	//! attachment number will index into the pAttachments array in the parent renderpass itself.
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	//! Depth attachment
	depth_attachment.flags = 0;
	depth_attachment.format = _depthFormat;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	
	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 1;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//! We are going to create 1 subpass, which is the minimum you can do.
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;
	//! Hook the depth attachment into the subpass
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	//! Create the array of two attachments
	VkAttachmentDescription attachments[2] = {
		color_attachment, depth_attachment
	};

	//! connect the color attachment to info
	render_pass_info.attachmentCount = 2;
	render_pass_info.pAttachments = attachments;
	//! connect the subpass to info
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _renderPass, nullptr);
	});
}

void VulkanEngine::init_framebuffers()
{
	//! create the framebuffers for the swapchain images. This will connect the renderpass to the images for rendering.
	VkFramebufferCreateInfo fb_info = {};
	fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	fb_info.pNext = nullptr;

	fb_info.renderPass = _renderPass;
	fb_info.attachmentCount = 1;
	fb_info.width = _windowExtent.width;
	fb_info.height = _windowExtent.height;
	fb_info.layers = 1;

	//! grap how many images we have in the swapchain
	const unsigned int swapchain_imagecount = static_cast<unsigned int>(_swapchainImages.size());
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

	//! create framebuffers for each of the swapchain image views
	for (unsigned int i = 0; i < swapchain_imagecount; ++i)
	{
		VkImageView image_views[2] = {
			_swapchainImageViews[i], _depthImageView
		};
		fb_info.attachmentCount = 2;
		fb_info.pAttachments = image_views;

		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
			vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
		});
	}
}

void VulkanEngine::init_sync_structures()
{
	//! create synchronization structures
	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_create_info.pNext = nullptr;

	//! We want to create the fence with the create signaled flag, so we can wait on it before using it on a GPU command(for the first frame).
	fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (auto& frame : _frames)
	{
		VK_CHECK(vkCreateFence(_device, &fence_create_info, nullptr, &frame._renderFence));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyFence(_device, frame._renderFence, nullptr);
		});

		//! for the semaphores, we dont need any flags.
		VkSemaphoreCreateInfo semaphore_create_info = {};
		semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		semaphore_create_info.pNext = nullptr;
		semaphore_create_info.flags = 0;

		VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &frame._renderSemaphore));
		VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &frame._presentSemaphore));

		_mainDeletionQueue.push_function([=]() {
			vkDestroySemaphore(_device, frame._renderSemaphore, nullptr);
			vkDestroySemaphore(_device, frame._presentSemaphore, nullptr);
		});
	}

	VkFenceCreateInfo uploadFenceInfo;
	uploadFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	uploadFenceInfo.pNext = nullptr;
	uploadFenceInfo.flags = 0;

	VK_CHECK(vkCreateFence(_device, &uploadFenceInfo, nullptr, &_uploadContext._uploadFence));
	_mainDeletionQueue.push_function([=]() {
		vkDestroyFence(_device, _uploadContext._uploadFence, nullptr);
	});
}

void VulkanEngine::init_pipelines()
{
	VkShaderModule triangleFragShader;
	if (!load_shader_module(RESOURCES_DIR "/shaders/triangle.frag.spv", &triangleFragShader))
	{
		std::cerr << "Error when building the triangle fragment shader module" << std::endl;
	}
	else
	{
		std::cout << "triangle fragment shader module successfully loaded" << std::endl;
	}

	VkShaderModule triangleVertShader;
	if (!load_shader_module(RESOURCES_DIR "/shaders/triangle.vert.spv", &triangleVertShader))
	{
		std::cerr << "Error when building the triangle vertex shader module" << std::endl;
	}
	else
	{
		std::cout << "triangle vertex shader module successfully loaded" << std::endl;
	}

	//! build the pipeline layout that controls the inputs/outputs of the shader.
	//! we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
	VkPipelineLayoutCreateInfo layout_info = vkinit::pipeline_layout_create_info();
	VK_CHECK(vkCreatePipelineLayout(_device, &layout_info, nullptr, &_trianglePipelineLayout));

	//! build the stage crate-info for both vertex and fragment stages. This let the pipeline know the shader modules per stage.
	PipelineBuilder pipeline_builder;

	pipeline_builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, triangleVertShader));
	pipeline_builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

	//! vertex input controls how to read vertices from vertex buffers. we aren't using it yet.
	pipeline_builder._vertexInputInfo = vkinit::pipeline_vertex_input_create_info();

	//! input assembly is the configuration for drawing triangle lists, strips, or individual points.
	//! we are just going to draw triangle list.
	pipeline_builder._inputAssemblyInfo = vkinit::pipeline_input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	//! build viewport and scissor from the swapchain extents.
	pipeline_builder._viewport.x = 0.0f;
	pipeline_builder._viewport.y = 0.0f;
	pipeline_builder._viewport.width  = static_cast<float>(_windowExtent.width);
	pipeline_builder._viewport.height = static_cast<float>(_windowExtent.height);
	pipeline_builder._viewport.minDepth = 0.0f;
	pipeline_builder._viewport.maxDepth = 1.0f;
	pipeline_builder._scissor.offset = { 0, 0 };
	pipeline_builder._scissor.extent = _windowExtent;

	//! configure the rasterizer to draw filled triangle
	pipeline_builder._rasterizationInfo = vkinit::pipeline_rasterization_create_info(VK_POLYGON_MODE_FILL);

	//! we dont use multisampling, so just run the default one.
	pipeline_builder._multisamplingInfo = vkinit::pipeline_multisample_create_info();

	//! a single blend attachment with no blending and writing to RGBA.
	pipeline_builder._colorBlendAttachment = vkinit::color_blend_attachment_state();

	//! default depth-testing
	pipeline_builder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

	//! use the triangle layout we created
	pipeline_builder._pipelineLayout = _trianglePipelineLayout;

	//! finally build the pipeline
	_trianglePipeline = pipeline_builder.build_pipeline(_device, _renderPass);

	//! clear the shader stages in the pipeline builder
	pipeline_builder._shaderStages.clear();

	VkShaderModule coloredTriangleFragShader;
	if (!load_shader_module(RESOURCES_DIR "/shaders/colored_triangle.frag.spv", &coloredTriangleFragShader))
	{
		std::cerr << "Error when building the triangle fragment shader module" << std::endl;
	}
	else
	{
		std::cout << "triangle fragment shader module successfully loaded" << std::endl;
	}

	VkShaderModule coloredTriangleVertShader;
	if (!load_shader_module(RESOURCES_DIR "/shaders/colored_triangle.vert.spv", &coloredTriangleVertShader))
	{
		std::cerr << "Error when building the triangle vertex shader module" << std::endl;
	}
	else
	{
		std::cout << "triangle vertex shader module successfully loaded" << std::endl;
	}

	//! push the new shader modules into shader stages
	pipeline_builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, coloredTriangleVertShader));
	pipeline_builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, coloredTriangleFragShader));

	//! build the one more pipeline
	_coloredTrianglePipeline = pipeline_builder.build_pipeline(_device, _renderPass);

	//! build the mesh pipeline
	VertexInputDescription vertexDescription = Vertex::get_vertex_description();
	//! connect the pipeline builder vertex input info to the one we get from vertex
	pipeline_builder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipeline_builder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();
	pipeline_builder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipeline_builder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	//! clear the shader stages in the pipeline builder
	pipeline_builder._shaderStages.clear();

	//! We start from just the default empty pipeline layout info
	VkPipelineLayoutCreateInfo mesh_pipeline_layout_create_info = vkinit::pipeline_layout_create_info();

	//! Setup push constants
	VkPushConstantRange push_constant;
	//! This push constants range starts at the beginning
	push_constant.offset = 0;
	//! This push constants range takes up the sizeof a MeshPushConstants struct
	push_constant.size = sizeof(MeshPushConstants);
	//! This push constant range is accessible only in the vertex shader.
	push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	mesh_pipeline_layout_create_info.pPushConstantRanges = &push_constant;
	mesh_pipeline_layout_create_info.pushConstantRangeCount = 1;

	//! hook the global set layout
	VkDescriptorSetLayout layouts[] = { _globalSetLayout, _objectSetLayout };
	mesh_pipeline_layout_create_info.pSetLayouts = layouts;
	mesh_pipeline_layout_create_info.setLayoutCount = 2;

	VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_create_info, nullptr, &_meshPipelineLayout));

	VkShaderModule defaultLitShader;
	if (!load_shader_module(RESOURCES_DIR "/shaders/default_lit.frag.spv", &defaultLitShader))
	{
		std::cerr << "Error when building the triangle fragment shader module" << std::endl;
	}
	else
	{
		std::cout << "triangle fragment shader module successfully loaded" << std::endl;
	}
	VkShaderModule meshVertShader;
	if (!load_shader_module(RESOURCES_DIR "/shaders/tri_mesh.vert.spv", &meshVertShader))
	{
		std::cerr << "Error when building the triangle fragment shader module" << std::endl;
	}
	else
	{
		std::cout << "triangle fragment shader module successfully loaded" << std::endl;
	}

	//! push the new shader modules into shader stages
	pipeline_builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));
	pipeline_builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, defaultLitShader));

	//! build the mesh triangle pipeline	
	pipeline_builder._pipelineLayout = _meshPipelineLayout;
	_meshPipeline = pipeline_builder.build_pipeline(_device, _renderPass);

	create_material(_meshPipeline, _meshPipelineLayout, "defaultmesh");

	VkShaderModule textureLitShader;
	if (!load_shader_module(RESOURCES_DIR "/shaders/texture_lit.frag.spv", &textureLitShader))
	{
		std::cerr << "Error when building the texture_lit fragment shader module" << std::endl;
	}
	else
	{
		std::cout << "texture_lit fragment shader module successfully loaded" << std::endl;
	}

	VkDescriptorSetLayout textured_layout[] = { _globalSetLayout, _objectSetLayout, _singleTextureLayout };

	VkPipelineLayoutCreateInfo textured_pipeline_layout_info = mesh_pipeline_layout_create_info;
	textured_pipeline_layout_info.setLayoutCount = 3;
	textured_pipeline_layout_info.pSetLayouts = textured_layout;

	VkPipelineLayout textured_pipeline_layout;
	VK_CHECK(vkCreatePipelineLayout(_device, &textured_pipeline_layout_info, nullptr, &textured_pipeline_layout));


	pipeline_builder._shaderStages.clear();
	pipeline_builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));
	pipeline_builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, textureLitShader));
	pipeline_builder._pipelineLayout = textured_pipeline_layout;

	VkPipeline texPipeline = pipeline_builder.build_pipeline(_device, _renderPass);
	create_material(texPipeline, textured_pipeline_layout, "texturedmesh");

	vkDestroyShaderModule(_device, coloredTriangleVertShader, nullptr);
	vkDestroyShaderModule(_device, coloredTriangleFragShader, nullptr);
	vkDestroyShaderModule(_device, triangleVertShader, nullptr);
	vkDestroyShaderModule(_device, triangleFragShader, nullptr);
	vkDestroyShaderModule(_device, meshVertShader, nullptr);
	vkDestroyShaderModule(_device, defaultLitShader, nullptr);

	_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(_device, _trianglePipeline, nullptr);
		vkDestroyPipeline(_device, _coloredTrianglePipeline, nullptr);
		vkDestroyPipeline(_device, _meshPipeline, nullptr);
		vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
		vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
	});
}

void VulkanEngine::init_scene()
{
	RenderObject monkey;
	monkey.material = get_material("defaultmesh");
	monkey.mesh = get_mesh("monkey");
	monkey.modelMatrix = glm::mat4(1.0f);

	_renderables.push_back(monkey);

	for (int x = -20; x <= 20; x++) 
	{
		for (int y = -20; y <= 20; y++) 
		{
			RenderObject tri;
			tri.mesh = get_mesh("triangle");
			tri.material = get_material("defaultmesh");
			glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x, 0, y));
			glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(0.2, 0.2, 0.2));
			tri.modelMatrix = translation * scale;

			_renderables.push_back(tri);
		}
	}

	RenderObject map;
	map.material = get_material("texturedmesh");
	map.mesh = get_mesh("empire");
	map.modelMatrix = glm::translate(glm::vec3{ 5,-10,0 });
	_renderables.push_back(map);

	VkSamplerCreateInfo sampler_info = vkinit::sampler_create_info(VK_FILTER_NEAREST);
	VkSampler blockySampler;
	vkCreateSampler(_device, &sampler_info, nullptr, &blockySampler);

	Material* texturedmesh = get_material("texturedmesh");

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.pNext = nullptr;
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = _descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &_singleTextureLayout;

	vkAllocateDescriptorSets(_device, &allocInfo, &texturedmesh->textureSet);

	VkDescriptorImageInfo imageBufferInfo = {};
	imageBufferInfo.sampler = blockySampler;
	imageBufferInfo.imageView = _loadedTextures["empire_diffuse"].imageView;
	imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkWriteDescriptorSet texture1 = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texturedmesh->textureSet, &imageBufferInfo, 0);
	vkUpdateDescriptorSets(_device, 1, &texture1, 0, nullptr);
}

void VulkanEngine::immeidate_submit(std::function<void(VkCommandBuffer)>&& function)
{
	//! allocate the default command buffer that we will use for the instant commands
	VkCommandBufferAllocateInfo allocInfo = vkinit::command_buffer_allocate_info(_uploadContext._commandPool, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(_device, &allocInfo, &cmd);

	//! Begin the command buffer recording. we will use this command buffer exactly once.
	//! so we want to let vulkan know about that
	VkCommandBufferBeginInfo cmdBeginInfo;
	cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBeginInfo.pNext = nullptr;
	cmdBeginInfo.pInheritanceInfo = nullptr;
	cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	//! execute the function
	function(cmd);

	VK_CHECK(vkEndCommandBuffer(cmd));
	VkSubmitInfo submitInfo;
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pNext = nullptr;
	submitInfo.pWaitDstStageMask = nullptr;
	submitInfo.pWaitSemaphores = nullptr;
	submitInfo.pSignalSemaphores = nullptr;
	submitInfo.signalSemaphoreCount = 0;
	submitInfo.waitSemaphoreCount = 0;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;

	//! Submit command buffet to the queue and execute it
	//! _uploadFence will now block until the graphic commands finish execution
	vkQueueSubmit(_graphicsQueue, 1, &submitInfo, _uploadContext._uploadFence);
	vkWaitForFences(_device, 1, &_uploadContext._uploadFence, true, 1e10);
	vkResetFences(_device, 1, &_uploadContext._uploadFence);

	//! Clear the command pool. This will free the command buffer too
	vkResetCommandPool(_device, _uploadContext._commandPool, 0);
}

bool VulkanEngine::load_shader_module(const char* filepath, VkShaderModule* outShaderModule)
{
	std::ifstream file(filepath, std::ios::ate | std::ios::binary);
	
	if (!file.is_open())
	{
		return false;
	}

	//! find the what the size of the file is by looking up the location of the cursor
	//! because the cursor is at the end, it gives the size directly by bytes.
	size_t fileSize = static_cast<size_t>(file.tellg());

	//! spirv expect the buffer to be on uint32, so make sure to reserve an int vector big enough for the entires file
	std::vector<unsigned int> buffer(fileSize / sizeof(unsigned int));

	//! put file cursor at beginning
	file.seekg(0);

	//! load the entire file into the buffer
	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

	//! now that file is loaded into the buffer, we can close it.
	file.close();

	//! create a new shader module, using the buffer we loaded.
	VkShaderModuleCreateInfo shaderCreateInfo = {};
	shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderCreateInfo.pNext = nullptr;

	//! code size has to be in bytes, so multiply the ints in the buffer by size of int to know real size of the buffer
	shaderCreateInfo.codeSize = buffer.size() * sizeof(unsigned int);
	shaderCreateInfo.pCode = buffer.data();

	//! check that the creation goes well
	VkShaderModule shaderModule;
	if (vkCreateShaderModule(_device, &shaderCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
	{
		return false;
	}

	*outShaderModule = shaderModule;
	return true;
}

void VulkanEngine::load_images()
{
	Texture lost_empire;
	vkutil::load_image_from_file(*this, RESOURCES_DIR "/textures/lost_empire-RGBA.png", lost_empire.image);

	VkImageViewCreateInfo viewInfo = vkinit::image_view_create_info(VK_FORMAT_R8G8B8A8_SRGB, lost_empire.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(_device, &viewInfo, nullptr, &lost_empire.imageView);

	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, lost_empire.imageView, nullptr);
	});

	_loadedTextures["empire_diffuse"] = lost_empire;
}

void VulkanEngine::load_meshes()
{
	//make the array 3 vertices long
	_triangleMesh._vertices.resize(3);

	//vertex positions
	_triangleMesh._vertices[0].position = { 1.f, 1.f, 0.0f };
	_triangleMesh._vertices[1].position = { -1.f, 1.f, 0.0f };
	_triangleMesh._vertices[2].position = { 0.f,-1.f, 0.0f };

	//vertex colors, all green
	_triangleMesh._vertices[0].color = { 0.f, 1.f, 0.0f }; //pure green
	_triangleMesh._vertices[1].color = { 0.f, 1.f, 0.0f }; //pure green
	_triangleMesh._vertices[2].color = { 0.f, 1.f, 0.0f }; //pure green

	//load the monkey
	_monkeyMesh.load_from_obj(RESOURCES_DIR "/objects/suzanne.obj");

	upload_mesh(_triangleMesh);
	upload_mesh(_monkeyMesh);

	_meshes["monkey"] = _monkeyMesh;
	_meshes["triangle"] = _triangleMesh;

	Mesh lostEmpire{};
	lostEmpire.load_from_obj(RESOURCES_DIR "/objects/lost_empire.obj");

	upload_mesh(lostEmpire);
	_meshes["empire"] = lostEmpire;
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t originalSize)
{
	//! Calculate required alignment based on minimum device offset alignment
	const size_t minUboAllignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
	size_t alignedSize = originalSize;
	if (minUboAllignment > 0)
		alignedSize = (alignedSize + minUboAllignment - 1) & ~(minUboAllignment - 1);
	return alignedSize;
}

void VulkanEngine::upload_mesh(Mesh& mesh)
{
	const size_t bufferSize = mesh._vertices.size() * sizeof(Vertex);
	//! Allocate staging buffer
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	//! Now VMA library knows that this data should be reside on CPU Ram.
	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
	
	AllocatedBuffer stagingBuffer;
	VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &stagingBuffer._buffer, &stagingBuffer._allocation, nullptr));

	void* data;
	vmaMapMemory(_allocator, stagingBuffer._allocation, &data);
	memcpy(data, mesh._vertices.data(), bufferSize);
	vmaUnmapMemory(_allocator, stagingBuffer._allocation);

	//! allocate the vertex buffer
	VkBufferCreateInfo buffer_info = {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.pNext = nullptr;
	//! this is the total size, in bytes, of the buffer we are allocating
	buffer_info.size = bufferSize;
	//! This buffer is going to be used as a vertex buffer.
	buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	//! Let the vma library know that this data should be writable by CPU, but also readable by GPU
	VmaAllocationCreateInfo vma_allocation_info = {};
	//! VMA_MEMORY_USAGE_CPU_TO_GPU useful for dynamic data
	vma_allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VK_CHECK(vmaCreateBuffer(_allocator, &buffer_info, &vma_allocation_info,	
							 &mesh._vertexBuffer._buffer, &mesh._vertexBuffer._allocation, nullptr));

	immeidate_submit([=](VkCommandBuffer cmd) {
		VkBufferCopy copy;
		copy.srcOffset = 0;
		copy.dstOffset = 0;
		copy.size = bufferSize;
		vkCmdCopyBuffer(cmd, stagingBuffer._buffer, mesh._vertexBuffer._buffer, 1, &copy);
	});

	//! Add the destruction of triangle mesh buffer to the deletion queue
	_mainDeletionQueue.push_function([=]() {
		vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
	});

	vmaDestroyBuffer(_allocator, stagingBuffer._buffer, stagingBuffer._allocation);
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
	//! allocate the vertex buffer
	VkBufferCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.pNext = nullptr;

	info.usage = usage;
	info.size = allocSize;

	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = memoryUsage;

	AllocatedBuffer newBuffer;

	//! Allocates the buffer
	VK_CHECK(vmaCreateBuffer(_allocator, &info, &allocInfo, &newBuffer._buffer, &newBuffer._allocation, nullptr));
	_mainDeletionQueue.push_function([=]() {
		vmaDestroyBuffer(_allocator, newBuffer._buffer, newBuffer._allocation);
	});

	return newBuffer;
}

void VulkanEngine::init_descriptors()
{
	//! Binding for camera data at 0.
	VkDescriptorSetLayoutBinding cameraBind = vkinit::descriptor_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	//! Binding for scene data at 1.
	VkDescriptorSetLayoutBinding sceneBind = vkinit::descriptor_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1);
	

	VkDescriptorSetLayoutBinding bindings[] = { cameraBind, sceneBind };

	VkDescriptorSetLayoutCreateInfo setInfo = {};
	setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setInfo.pNext = nullptr;
	setInfo.pBindings = bindings;
	setInfo.bindingCount = 2;
	setInfo.flags = 0;

	VK_CHECK(vkCreateDescriptorSetLayout(_device, &setInfo, nullptr, &_globalSetLayout));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);
	});

	//! Create a descriptor pool that will hold 10 uniform buffers
	std::vector<VkDescriptorPoolSize> sizes = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 }
	};

	VkDescriptorPoolCreateInfo poolCreateInfo = {};
	poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolCreateInfo.pNext = nullptr;
	poolCreateInfo.flags = 0;
	poolCreateInfo.maxSets = 10;
	poolCreateInfo.poolSizeCount = static_cast<unsigned int>(sizes.size());
	poolCreateInfo.pPoolSizes = sizes.data();

	VK_CHECK(vkCreateDescriptorPool(_device, &poolCreateInfo, nullptr, &_descriptorPool));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
	});

	const size_t sceneParamBufferSize = kFrameOverlap * pad_uniform_buffer_size(sizeof(GPUSceneData));
	_sceneParameterBuffer = create_buffer(sceneParamBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	VkDescriptorSetLayoutBinding objectLayout = vkinit::descriptor_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	VkDescriptorSetLayoutCreateInfo objectLayoutInfo = {};
	objectLayoutInfo.bindingCount = 1;
	objectLayoutInfo.pBindings = &objectLayout;
	objectLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	objectLayoutInfo.pNext = nullptr;
	objectLayoutInfo.flags = 0;

	VK_CHECK(vkCreateDescriptorSetLayout(_device, &objectLayoutInfo, nullptr, &_objectSetLayout));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(_device, _objectSetLayout, nullptr);
	});

	for (int i = 0; i < kFrameOverlap; ++i)
	{
		const int maxObjects = 10000;
		_frames[i].objectBuffer = create_buffer(sizeof(GPUObjectData) * maxObjects, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		_frames[i].cameraBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		//! allocates one descriptor set for each frame
		VkDescriptorSetAllocateInfo setAllocInfo = {};
		setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		setAllocInfo.pNext = nullptr;
		//! using the pool we just set
		setAllocInfo.descriptorPool = _descriptorPool;
		//! Using the global data layout;
		setAllocInfo.pSetLayouts = &_globalSetLayout;
		//! Only one descriptor
		setAllocInfo.descriptorSetCount = 1;

		vkAllocateDescriptorSets(_device, &setAllocInfo, &_frames[i].globalDescriptor);

		//! allocates one descriptor set for each frame
		VkDescriptorSetAllocateInfo objectSetAlloc = {};
		objectSetAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		objectSetAlloc.pNext = nullptr;
		//! using the pool we just set
		objectSetAlloc.descriptorPool = _descriptorPool;
		//! Using the global data layout;
		objectSetAlloc.pSetLayouts = &_objectSetLayout;
		//! Only one descriptor
		objectSetAlloc.descriptorSetCount = 1;

		vkAllocateDescriptorSets(_device, &objectSetAlloc, &_frames[i].objectDescriptor);
		
		//! Information about the buffer we want to point at in the desciptor.
		VkDescriptorBufferInfo bufferInfo = {};
		//! It will be the camera buffer
		bufferInfo.buffer = _frames[i].cameraBuffer._buffer;
		//! at 0 offset
		bufferInfo.offset = 0;
		//! of the size of a camera data struct
		bufferInfo.range = sizeof(GPUCameraData);

		VkDescriptorBufferInfo sceneInfo;
		sceneInfo.buffer = _sceneParameterBuffer._buffer;
		sceneInfo.offset = 0;
		sceneInfo.range = sizeof(GPUSceneData);

		VkDescriptorBufferInfo objectInfo;
		objectInfo.buffer = _frames[i].objectBuffer._buffer;
		objectInfo.offset = 0;
		objectInfo.range = sizeof(GPUObjectData) * maxObjects;

		VkWriteDescriptorSet camWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].globalDescriptor, &bufferInfo, 0);
		VkWriteDescriptorSet sceneWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, _frames[i].globalDescriptor, &sceneInfo, 1);
		VkWriteDescriptorSet objectWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _frames[i].objectDescriptor, &objectInfo, 0);

		VkWriteDescriptorSet setWrites[] = { camWrite, sceneWrite, objectWrite };
		vkUpdateDescriptorSets(_device, 3, setWrites, 0, nullptr);
	}

	//! Binding for texture at 2
	VkDescriptorSetLayoutBinding textureBind = vkinit::descriptor_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	VkDescriptorSetLayoutCreateInfo textureInfo = {};
	textureInfo.bindingCount = 1;
	textureInfo.pBindings = &textureBind;
	textureInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	textureInfo.pNext = nullptr;
	textureInfo.flags = 0;

	vkCreateDescriptorSetLayout(_device, &textureInfo, nullptr, &_singleTextureLayout);
}

Material* VulkanEngine::create_material(VkPipeline pipeline, VkPipelineLayout pipelineLayout, const std::string& name)
{
	Material mat;
	mat.pipeline = pipeline;
	mat.pipelineLayout = pipelineLayout;
	_materials[name] = mat;
	return &_materials[name];
}

Material* VulkanEngine::get_material(const std::string& name)
{
	//search for the object, and return nullpointer if not found
	auto it = _materials.find(name);
	if (it == _materials.end()) {
		return nullptr;
	}
	else {
		return &(*it).second;
	}
}

Mesh* VulkanEngine::get_mesh(const std::string& name)
{
	//search for the object, and return nullpointer if not found
	auto it = _meshes.find(name);
	if (it == _meshes.end()) {
		return nullptr;
	}
	else {
		return &(*it).second;
	}
}

void VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject* first, int count)
{
	//! make a model view matrix for rendering the object.
	//! camera view
	glm::vec3 camPos = { 0.0f, -6.0f, -10.0f };
	glm::mat4 view = glm::translate(glm::mat4(1.0f), camPos);
	glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1700.0f / 900.0f, 0.1f, 200.0f);
	projection[1][1] *= -1;

	//! GPU Camera data
	GPUCameraData camData;
	camData.projection = projection;
	camData.view = view;
	camData.viewProj = projection * view;

	//! And copy it to the buffer
	void* data;
	vmaMapMemory(_allocator, get_current_frame().cameraBuffer._allocation, &data);
	memcpy(data, &camData, sizeof(GPUCameraData));
	vmaUnmapMemory(_allocator, get_current_frame().cameraBuffer._allocation);

	//! GPU Scene data
	float framed = _frameNumber / 120.0f;
	_sceneParameters.ambientColor = { sin(framed), 0.0f, cos(framed), 1.0f };
	char* sceneData;
	vmaMapMemory(_allocator, _sceneParameterBuffer._allocation, reinterpret_cast<void**>(&sceneData));
	int frameIndex = _frameNumber % 2;
	sceneData += pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;
	memcpy(sceneData, &_sceneParameters, sizeof(GPUSceneData));
	vmaUnmapMemory(_allocator, _sceneParameterBuffer._allocation);

	void* objectData;
	vmaMapMemory(_allocator, get_current_frame().objectBuffer._allocation, &objectData);
	GPUObjectData* objectSSBO = reinterpret_cast<GPUObjectData*>(objectData);

	for (int i = 0; i < count; ++i)
	{
		RenderObject& object = first[i];
		objectSSBO[i].modelMatrix = object.modelMatrix;
	}

	vmaUnmapMemory(_allocator, get_current_frame().objectBuffer._allocation);

	Mesh* last_mesh = nullptr;
	Material* last_material = nullptr;

	for (int i = 0; i < count; ++i)
	{
		RenderObject& obj = first[i];

		//! Only bind the pipeline if it does not match with the already bound one
		if (obj.material != last_material)
		{
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, obj.material->pipeline);
			last_material = obj.material;
			//! Bind the descriptor set when changing pipeline
			unsigned int offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, obj.material->pipelineLayout,
									0, 1, &get_current_frame().globalDescriptor, 1, &offset);
			//! Bind the object descriptor set
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, obj.material->pipelineLayout, 1,
				1, &get_current_frame().objectDescriptor, 0, nullptr);

			if (obj.material->textureSet != VK_NULL_HANDLE)
			{
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, obj.material->pipelineLayout, 2, 1, &obj.material->textureSet, 0, nullptr);
			}
		}

		MeshPushConstants pushConstants;
		pushConstants.render_matrix = obj.modelMatrix;

		vkCmdPushConstants(cmd, obj.material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &pushConstants);

		//! only bind the mesh if it is a different one from last bind
		if (obj.mesh != last_mesh)
		{
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &obj.mesh->_vertexBuffer._buffer, &offset);
			last_mesh = obj.mesh;
		}
		vkCmdDraw(cmd, obj.mesh->_vertices.size(), 1, 0, i);
	}
}

FrameData& VulkanEngine::get_current_frame()
{
	return _frames[_frameNumber % kFrameOverlap];
}

void VulkanEngine::cleanup()
{	
	// Queue and PhysicalDevice cannot be destroyed
	// because they are not really created objects in this application.
	if (_isInitialized) {
		//make sure the GPU has stopped doing its things
		vkDeviceWaitIdle(_device);

		_mainDeletionQueue.flush();

		vkDestroySurfaceKHR(_instance, _surface, nullptr);

		vkDestroyDevice(_device, nullptr);
		vkb::destroy_debug_utils_messenger(_instance, _debug_messenger, nullptr);
		vkDestroyInstance(_instance, nullptr);

		SDL_DestroyWindow(_window);
	}
}

void VulkanEngine::draw()
{
	auto& current_frame = get_current_frame();
	//! wait until the GPU has finished rendering the last frame. Timeout of 1 second (unit : nanoseconds 1e-9)
	VK_CHECK(vkWaitForFences(_device, 1, &current_frame._renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_device, 1, &current_frame._renderFence));

	//! request image from the swapchain, one second timeout
	unsigned int swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, current_frame._presentSemaphore, nullptr, &swapchainImageIndex));

	//! now that we are sure that the commands finished executing, we can safely reset the command buffer to begin reordering again
	VK_CHECK(vkResetCommandBuffer(current_frame._mainCommandBuffer, 0));

	//! naming it cmd for shorter writing
	VkCommandBuffer cmd = current_frame._mainCommandBuffer;

	//! begin the command buffer reordering. we will use this command buffer exactly once, so we want to let vulkan know that.
	VkCommandBufferBeginInfo cmd_begin_info = {};
	cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmd_begin_info.pNext = nullptr;
	cmd_begin_info.pInheritanceInfo = nullptr;
	cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

	//! make a clear-color from frame number. This will flash with a 120*pi frame period.
	VkClearValue clearValue;
	float flash = abs(sin(_frameNumber / 120.0f));
	clearValue.color = { {0.0f, 0.0f, flash, 1.0f} };

	//! Clear depth at 1
	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.0f;

	//! start the main renderpass
	//! we will use the clear color from above, and the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo rpInfo = {};
	rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpInfo.pNext = nullptr;

	rpInfo.renderPass = _renderPass;
	rpInfo.renderArea.offset.x = 0;
	rpInfo.renderArea.offset.y = 0;
	rpInfo.renderArea.extent = _windowExtent;
	rpInfo.framebuffer = _framebuffers[swapchainImageIndex];

	//! connect clear values
	rpInfo.clearValueCount = 2;

	VkClearValue clearValues[] = { clearValue, depthClear };
	rpInfo.pClearValues = clearValues;

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	draw_objects(cmd, _renderables.data(), _renderables.size());

	//! finalize the render pass
	vkCmdEndRenderPass(cmd);

	//! finalize the command buffer (we can no longer add commands, but it can now be executed)
	VK_CHECK(vkEndCommandBuffer(cmd));

	//! prepare the submission to the queue
	//! we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready.
	//! we will signal the _renderSemaphore, to signal that rendering has finished
	VkSubmitInfo submit = {};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext = nullptr;

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;
	
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &current_frame._presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &current_frame._renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;
	
	//! submit command buffer to the queue and execute it.
	//! _renderFence will now block until the graphics commands finish execution
	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, current_frame._renderFence));

	//! This will put the image we just rendered into the visible window.
	//! we want to wait on the _renderSemaphore for that
	//! as it's necessary that drawing commands have finished before the image is displayed to the user.
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &_swapchain;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &current_frame._renderSemaphore;
	
	presentInfo.pImageIndices = &swapchainImageIndex;

	VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

	//! increate the number of frames drawn
	++_frameNumber;
}

void VulkanEngine::run()
{
	SDL_Event e;
	bool bQuit = false;

	//main loop
	while (!bQuit)
	{
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			//close the window when user alt-f4s or clicks the X button			
			if (e.type == SDL_QUIT) 
				bQuit = true;	
			else if (e.type == SDL_KEYDOWN)
			{
				if (e.key.keysym.sym == SDLK_SPACE)
				{
					_selectedShader += 1;
					if (_selectedShader > 1)
					{
						_selectedShader = 0;
					}
				}
				else if (e.key.keysym.sym == SDLK_ESCAPE)
					bQuit = true;
			}
		}

		draw();
	}
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass renderPass)
{
	//! make viewport state from our stored viewport and scissor
	//! at the moment we won't support multiple viewports or scissors
	VkPipelineViewportStateCreateInfo viewport_info = {};
	viewport_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_info.pNext = nullptr;

	viewport_info.viewportCount = 1;
	viewport_info.pViewports = &_viewport;
	viewport_info.scissorCount = 1;
	viewport_info.pScissors = &_scissor;

	//! setup dummy color blending. we aren't using transparent objects yet.
	//! the blending is just "no blend", but we do writing to the color attachment.
	VkPipelineColorBlendStateCreateInfo color_blending_info = {};
	color_blending_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blending_info.pNext = nullptr;

	color_blending_info.logicOpEnable = VK_FALSE;
	color_blending_info.logicOp = VK_LOGIC_OP_COPY;
	color_blending_info.attachmentCount = 1;
	color_blending_info.pAttachments = &_colorBlendAttachment;

	//! build the actual pipeline
	//! we now use all of the info structures we have been writing into this one to create the pipeline.
	VkGraphicsPipelineCreateInfo pipeline_info = {};
	pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline_info.pNext = nullptr;

	pipeline_info.stageCount = _shaderStages.size();
	pipeline_info.pStages = _shaderStages.data();
	pipeline_info.pVertexInputState = &_vertexInputInfo;
	pipeline_info.pInputAssemblyState = &_inputAssemblyInfo;
	pipeline_info.pViewportState = &viewport_info;
	pipeline_info.pRasterizationState = &_rasterizationInfo;
	pipeline_info.pMultisampleState = &_multisamplingInfo;
	pipeline_info.pColorBlendState = &color_blending_info;
	pipeline_info.pDepthStencilState = &_depthStencil;
	pipeline_info.layout = _pipelineLayout;
	pipeline_info.renderPass = renderPass;
	pipeline_info.subpass = 0;
	pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

	//! it is easy to error out on create graphics pipeline, so we handle it a bit better than the VK_CHECK case
	VkPipeline pipeline;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
	{
		std::cerr << "Failed to create pipeline" << std::endl;
		return VK_NULL_HANDLE;
	}
	else
	{
		return pipeline;
	}
}