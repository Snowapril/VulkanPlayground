#include <vk_engine.h>

int main(int argc, char* argv[])
{
	(void)argc; (void)argv;
	VulkanEngine engine;

	engine.init();	
	
	engine.run();	

	engine.cleanup();	

	return 0;
}
