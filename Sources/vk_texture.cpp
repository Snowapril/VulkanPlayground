#include <vk_texture.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <vk_initializers.h>

#include <stb_image/stb_image.h>
bool vkutil::load_image_from_file(VulkanEngine& engine, const char* file, AllocatedImage& outImage)
{
	int width, height, numChannels;

	//! STBI_rgb_alpha flag makes library always load image as RGBA format
	stbi_uc* pixels = stbi_load(file, &width, &height, &numChannels, STBI_rgb_alpha);
	if (pixels == nullptr)
	{
		std::cerr << "Failed to load texture file : " << file << std::endl;
		return false;
	}

	void* pixel_ptr = pixels;
	VkDeviceSize textureSize = width * height * 4;
	//! the format R8G8B8A8 matches exactly with the pixels loaded from stb_image lib
	VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
	//! allocate temporary buffer for holding texture data to upload
	AllocatedBuffer stagingBuffer = engine.create_buffer(textureSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

	void* data;
	vmaMapMemory(engine._allocator, stagingBuffer._allocation, &data);
	memcpy(data, pixel_ptr, static_cast<size_t>(textureSize));
	vmaUnmapMemory(engine._allocator, stagingBuffer._allocation);

	stbi_image_free(pixels);

	VkExtent3D imageExtent;
	imageExtent.width  = static_cast<unsigned int>(width);
	imageExtent.height = static_cast<unsigned int>(height);
	imageExtent.depth  = static_cast<unsigned int>(1);

	VkImageCreateInfo imageInfo = vkinit::image_create_info(format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, imageExtent);
	AllocatedImage newImage;
	
	VmaAllocationCreateInfo dimg_alloc_info = {};
	dimg_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	vmaCreateImage(engine._allocator, &imageInfo, &dimg_alloc_info, &newImage._image, &newImage._allocation, nullptr);

	engine.immeidate_submit([=](VkCommandBuffer cmd) {
		VkImageSubresourceRange range;
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;

		VkImageMemoryBarrier imageBarrier_toTransfer = {};
		imageBarrier_toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageBarrier_toTransfer.pNext = nullptr;

		imageBarrier_toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarrier_toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toTransfer.image = newImage._image;
		imageBarrier_toTransfer.subresourceRange = range;
		imageBarrier_toTransfer.srcAccessMask = 0;
		imageBarrier_toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		//! barrier the image into the transfer-receive layout
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toTransfer);

		VkBufferImageCopy copyRegion = {};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferImageHeight = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageExtent = imageExtent;

		//! Copy the buffer into the image
		vkCmdCopyBufferToImage(cmd, stagingBuffer._buffer, newImage._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		imageBarrier_toTransfer.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toTransfer.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageBarrier_toTransfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageBarrier_toTransfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		//! Barrier the image into the shader readable layout
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toTransfer);
	});

	engine._mainDeletionQueue.push_function([=]() {
		vmaDestroyImage(engine._allocator, newImage._image, newImage._allocation);
	});
	vmaDestroyBuffer(engine._allocator, stagingBuffer._buffer, stagingBuffer._allocation);

	std::cout << "Texture loading success" << std::endl;
	outImage = newImage;

	return true;
}