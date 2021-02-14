// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

namespace vkinit {

	//vulkan init code goes here
	VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool commandPool, unsigned int commandBufferCount, VkCommandBufferLevel level);
	VkCommandPoolCreateInfo command_pool_create_info(unsigned int familyIndex, VkCommandPoolCreateFlags flags = 0);

	VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(VkShaderStageFlagBits stage, VkShaderModule shaderModule);
	VkPipelineVertexInputStateCreateInfo pipeline_vertex_input_create_info();
	VkPipelineInputAssemblyStateCreateInfo pipeline_input_assembly_create_info(VkPrimitiveTopology topology);
	VkPipelineRasterizationStateCreateInfo pipeline_rasterization_create_info(VkPolygonMode mode);
	VkPipelineMultisampleStateCreateInfo pipeline_multisample_create_info();
	VkPipelineColorBlendAttachmentState color_blend_attachment_state();

	VkPipelineLayoutCreateInfo pipeline_layout_create_info();
};