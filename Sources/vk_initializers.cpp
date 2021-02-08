#include <vk_initializers.h>

namespace vkinit {
	VkCommandPoolCreateInfo command_pool_create_info(unsigned int familyIndex, VkCommandPoolCreateFlags flags)
	{
		// create a command pool for commands submitted to the graphics queue
		VkCommandPoolCreateInfo commandPoolInfo = {};
		commandPoolInfo.pNext = nullptr;
		commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

		// the command pool will be one that can submit graphics commands
		commandPoolInfo.queueFamilyIndex = familyIndex;
		// we also want the pool to allow for resetting of individual commands buffers
		commandPoolInfo.flags = flags;

		return commandPoolInfo;
	}

	VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool commandPool, unsigned int commandBufferCount, VkCommandBufferLevel level)
	{
		// allocate the default command buffer that we will use for rendering
		VkCommandBufferAllocateInfo commandBufferInfo = {};
		commandBufferInfo.pNext = nullptr;
		commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

		// command buffers will be allocated from our command pool
		commandBufferInfo.commandPool = commandPool;
		// we will allocate 1 command buffer
		commandBufferInfo.commandBufferCount = commandBufferCount;
		// command level is primary
		commandBufferInfo.level = level;

		return commandBufferInfo;
	}
};