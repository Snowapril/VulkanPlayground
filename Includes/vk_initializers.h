// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

namespace vkinit {

	//vulkan init code goes here
	VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool commandPool, unsigned int commandBufferCount, VkCommandBufferLevel level);
	VkCommandPoolCreateInfo command_pool_create_info(unsigned int familyIndex, VkCommandPoolCreateFlags flags = 0);
}

