﻿// vulkan_guide.h : Include file for standard system include files,
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

	VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent);
	VkImageViewCreateInfo image_view_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags);
	VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info(bool bDepthTest, bool bDepthWrite, VkCompareOp compareOp);

	VkDescriptorSetLayoutBinding descriptor_layout_binding(VkDescriptorType type, VkShaderStageFlags stageFlags, unsigned int binding);
	VkWriteDescriptorSet write_descriptor_buffer(VkDescriptorType type, VkDescriptorSet dstSet, VkDescriptorBufferInfo* bufferInfo, unsigned int binding);

	VkSamplerCreateInfo sampler_create_info(VkFilter filters, VkSamplerAddressMode samplerAddressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT);
	VkWriteDescriptorSet write_descriptor_image(VkDescriptorType type, VkDescriptorSet dstSet, VkDescriptorImageInfo* imageInfo, unsigned int binding);
};