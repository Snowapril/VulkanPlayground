#include <vk_initializers.h>

namespace vkinit {
	VkCommandPoolCreateInfo command_pool_create_info(unsigned int familyIndex, VkCommandPoolCreateFlags flags)
	{
		//! create a command pool for commands submitted to the graphics queue
		VkCommandPoolCreateInfo commandPoolInfo = {};
		commandPoolInfo.pNext = nullptr;
		commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

		//! the command pool will be one that can submit graphics commands
		commandPoolInfo.queueFamilyIndex = familyIndex;
		//! we also want the pool to allow for resetting of individual commands buffers
		commandPoolInfo.flags = flags;

		return commandPoolInfo;
	}

	VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool commandPool, unsigned int commandBufferCount, VkCommandBufferLevel level)
	{
		//! allocate the default command buffer that we will use for rendering
		VkCommandBufferAllocateInfo commandBufferInfo = {};
		commandBufferInfo.pNext = nullptr;
		commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

		//! command buffers will be allocated from our command pool
		commandBufferInfo.commandPool = commandPool;
		//! we will allocate 1 command buffer
		commandBufferInfo.commandBufferCount = commandBufferCount;
		//! command level is primary
		commandBufferInfo.level = level;

		return commandBufferInfo;
	}
	
	VkPipelineShaderStageCreateInfo vkinit::pipeline_shader_stage_create_info(VkShaderStageFlagBits stage, VkShaderModule shaderModule)
	{
		VkPipelineShaderStageCreateInfo shader_info = {};
		shader_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shader_info.pNext = nullptr;

		shader_info.stage = stage;
		shader_info.module = shaderModule;

		//! the entry point of the module.
		shader_info.pName = "main";

		return shader_info;
	}

	VkPipelineVertexInputStateCreateInfo vkinit::pipeline_vertex_input_create_info()
	{
		VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
		vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_info.pNext = nullptr;

		//! no vertex bindings or attributes.
		vertex_input_info.vertexBindingDescriptionCount = 0;
		vertex_input_info.vertexAttributeDescriptionCount = 0;

		return vertex_input_info;
	}
	
	VkPipelineInputAssemblyStateCreateInfo vkinit::pipeline_input_assembly_create_info(VkPrimitiveTopology topology)
	{
		VkPipelineInputAssemblyStateCreateInfo input_assembly_info = {};
		input_assembly_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly_info.pNext = nullptr;

		input_assembly_info.topology = topology;
		//! we are not going to use primitive restart on the entire tutorial so leave it as false
		input_assembly_info.primitiveRestartEnable = VK_FALSE;
		return input_assembly_info;
	}

	VkPipelineRasterizationStateCreateInfo vkinit::pipeline_rasterization_create_info(VkPolygonMode mode)
	{
		VkPipelineRasterizationStateCreateInfo rasterization_info = {};
		rasterization_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterization_info.pNext = nullptr;

		rasterization_info.polygonMode = mode;
		rasterization_info.depthBiasEnable = VK_FALSE;
		//! discards all primitives before the rasterization stage if enabled which we dont want.
		rasterization_info.rasterizerDiscardEnable = VK_FALSE;
		rasterization_info.lineWidth = 1.0f;
		//! no backface cull
		rasterization_info.cullMode = VK_CULL_MODE_NONE;
		rasterization_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		//! no depth bias
		rasterization_info.depthBiasEnable = VK_FALSE;
		rasterization_info.depthBiasConstantFactor = 0.0f;
		rasterization_info.depthBiasClamp = 0.0f;
		rasterization_info.depthBiasSlopeFactor = 0.0f;

		return rasterization_info;
	}

	VkPipelineMultisampleStateCreateInfo vkinit::pipeline_multisample_create_info()
	{
		VkPipelineMultisampleStateCreateInfo multisample_info = {};
		multisample_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample_info.pNext = nullptr;

		multisample_info.sampleShadingEnable = VK_FALSE;
		//! multisampling defaulted to no multisampling (1 sample per pixel)
		multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisample_info.minSampleShading = 1.0f;
		multisample_info.pSampleMask = nullptr;
		multisample_info.alphaToCoverageEnable = VK_FALSE;
		multisample_info.alphaToOneEnable = VK_FALSE;

		return multisample_info;
	}
	
	VkPipelineColorBlendAttachmentState vkinit::color_blend_attachment_state()
	{
		VkPipelineColorBlendAttachmentState color_blend_attachment = {};
		color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
												VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		color_blend_attachment.blendEnable = VK_FALSE;
		
		return color_blend_attachment;
	}

	VkPipelineLayoutCreateInfo vkinit::pipeline_layout_create_info()
	{
		VkPipelineLayoutCreateInfo layout_info = {};
		layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layout_info.pNext = nullptr;

		//! empty defaults
		layout_info.flags = 0;
		layout_info.setLayoutCount = 0;
		layout_info.pSetLayouts = nullptr;
		layout_info.pushConstantRangeCount = 0;
		layout_info.pPushConstantRanges = nullptr;

		return layout_info;
	}

	VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent)
	{
		VkImageCreateInfo image_info = {};
		image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image_info.pNext = nullptr;

		image_info.imageType = VK_IMAGE_TYPE_2D;
		image_info.format = format;
		image_info.extent = extent;

		image_info.mipLevels = 1;
		image_info.arrayLayers = 1;
		image_info.samples = VK_SAMPLE_COUNT_1_BIT;
		image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
		image_info.usage = usageFlags;

		return image_info;
	}

	VkImageViewCreateInfo vkinit::image_view_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags)
	{
		VkImageViewCreateInfo image_view_info = {};
		image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		image_view_info.pNext = nullptr;

		image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		image_view_info.image = image;
		image_view_info.format = format;
		image_view_info.subresourceRange.baseArrayLayer = 0;
		image_view_info.subresourceRange.baseMipLevel = 0;
		image_view_info.subresourceRange.levelCount = 1;
		image_view_info.subresourceRange.layerCount = 1;
		image_view_info.subresourceRange.aspectMask = aspectFlags;

		return image_view_info;
	}
	
	VkPipelineDepthStencilStateCreateInfo vkinit::depth_stencil_create_info(bool bDepthTest, bool bDepthWrite, VkCompareOp compareOp)
	{
		VkPipelineDepthStencilStateCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		info.pNext = nullptr;

		info.depthTestEnable = bDepthTest;
		info.depthWriteEnable = bDepthWrite;
		info.depthCompareOp = bDepthTest ? compareOp : VK_COMPARE_OP_ALWAYS;
		info.minDepthBounds = 0.0f;
		info.maxDepthBounds = 1.0f;
		info.stencilTestEnable = VK_FALSE;
		
		return info;
	}

	VkDescriptorSetLayoutBinding vkinit::descriptor_layout_binding(VkDescriptorType type, VkShaderStageFlags stageFlags, unsigned int binding)
	{
		VkDescriptorSetLayoutBinding setBind = {};
		setBind.stageFlags = stageFlags;
		setBind.binding = binding;
		setBind.descriptorType = type;
		setBind.descriptorCount = 1;
		setBind.pImmutableSamplers = nullptr;

		return setBind;
	}

	VkWriteDescriptorSet vkinit::write_descriptor_buffer(VkDescriptorType type, VkDescriptorSet dstSet, VkDescriptorBufferInfo* bufferInfo, unsigned int binding)
	{
		VkWriteDescriptorSet writeSet = {};
		writeSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeSet.pNext = nullptr;

		writeSet.dstBinding = binding;
		writeSet.dstSet = dstSet;
		writeSet.descriptorCount = 1;
		writeSet.descriptorType = type;
		writeSet.pBufferInfo = bufferInfo;

		return writeSet;
	}
};