
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>

#include <VkBootstrap.h>

#include <iostream>
#include <fstream>

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

	//init pipelines
	init_pipelines();

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

	// use vkbootstrap to grabbing the queue
	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
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
}

void VulkanEngine::init_commands()
{
	// create a command pool for commands submitted into the graphics queue
	// we also want the pool to allow for resetting of individual command buffers
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

	// allocate the default buffer that will be used for rendering
	VkCommandBufferAllocateInfo commandBufferInfo = vkinit::command_buffer_allocate_info(_commandPool, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	VK_CHECK(vkAllocateCommandBuffers(_device, &commandBufferInfo, &_commandBuffer));
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

	//! We are going to create 1 subpass, which is the minimum you can do.
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	//! connect the color attachment to info
	render_pass_info.attachmentCount = 1;
	render_pass_info.pAttachments = &color_attachment;
	//! connect the subpass to info
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));
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
		fb_info.pAttachments = &_swapchainImageViews[i];
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));
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

	VK_CHECK(vkCreateFence(_device, &fence_create_info, nullptr, &_renderFence));
	
	//! for the semaphores, we dont need any flags.
	VkSemaphoreCreateInfo semaphore_create_info = {};
	semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphore_create_info.pNext = nullptr;
	semaphore_create_info.flags = 0;

	VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &_renderSemaphore));
	VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &_presentSemaphore));
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

	//! use the triangle layout we created
	pipeline_builder._pipelineLayout = _trianglePipelineLayout;

	//! finally build the pipeline
	_trianglePipeline = pipeline_builder.build_pipeline(_device, _renderPass);

	vkDestroyShaderModule(_device, triangleVertShader, nullptr);
	vkDestroyShaderModule(_device, triangleFragShader, nullptr);
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

void VulkanEngine::cleanup()
{	
	// Queue and PhysicalDevice cannot be destroyed
	// because they are not really created objects in this application.
	if (_isInitialized) {
		vkDeviceWaitIdle(_device);
		
		vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
		vkDestroyPipeline(_device, _trianglePipeline, nullptr);

		vkDestroyFence(_device, _renderFence, nullptr);
		vkDestroySemaphore(_device, _renderSemaphore, nullptr);
		vkDestroySemaphore(_device, _presentSemaphore, nullptr);

		vkDestroyCommandPool(_device, _commandPool, nullptr);

		vkDestroySwapchainKHR(_device, _swapchain, nullptr);		

		// Destroy the main renderpass
		vkDestroyRenderPass(_device, _renderPass, nullptr);

		// no need to destroy vkImage because 
		// it is owned and destroyed by swapchain.
		for (unsigned int i = 0; i < _framebuffers.size(); ++i)
		{
			vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
			vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
		}

		vkDestroyDevice(_device, nullptr);
		vkDestroySurfaceKHR(_instance, _surface, nullptr);
		vkb::destroy_debug_utils_messenger(_instance, _debug_messenger, nullptr);
		vkDestroyInstance(_instance, nullptr);
		SDL_DestroyWindow(_window);
	}
}

void VulkanEngine::draw()
{
	//! wait until the GPU has finished rendering the last frame. Timeout of 1 second (unit : nanoseconds 1e-9)
	VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_device, 1, &_renderFence));

	//! request image from the swapchain, one second timeout
	unsigned int swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, _presentSemaphore, nullptr, &swapchainImageIndex));

	//! now that we are sure that the commands finished executing, we can safely reset the command buffer to begin reordering again
	VK_CHECK(vkResetCommandBuffer(_commandBuffer, 0));

	//! naming it cmd for shorter writing
	VkCommandBuffer cmd = _commandBuffer;

	//! begin the command buffer reordering. we will use this command buffer exactly once, so we want to let vulkan know that.
	VkCommandBufferBeginInfo cmd_begin_info = {};
	cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmd_begin_info.pNext = nullptr;
	cmd_begin_info.pInheritanceInfo = nullptr;
	cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(_commandBuffer, &cmd_begin_info));

	//! make a clear-color from frame number. This will flash with a 120*pi frame period.
	VkClearValue clearValue;
	float flash = abs(sin(_frameNumber / 120.0f));
	clearValue.color = { {0.0f, 0.0f, flash, 1.0f} };

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
	rpInfo.clearValueCount = 1;
	rpInfo.pClearValues = &clearValue;

	vkCmdBeginRenderPass(_commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _trianglePipeline);
	vkCmdDraw(_commandBuffer, 3, 1, 0, 0);

	//! finalize the render pass
	vkCmdEndRenderPass(_commandBuffer);
	//! finalize the command buffer (we can no longer add commands, but it can now be executed)
	VK_CHECK(vkEndCommandBuffer(_commandBuffer));

	//! prepare the submission to the queue
	//! we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready.
	//! we will signal the _renderSemaphore, to signal that rendering has finished
	VkSubmitInfo submit = {};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext = nullptr;

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;
	
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &_presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &_renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;
	
	//! submit command buffer to the queue and execute it.
	//! _renderFence will now block until the graphics commands finish execution
	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

	//! This will put the image we just rendered into the visible window.
	//! we want to wait on the _renderSemaphore for that
	//! as it's necessary that drawing commands have finished before the image is displayed to the user.
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &_swapchain;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &_renderSemaphore;

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
			if (e.type == SDL_QUIT) bQuit = true;	
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