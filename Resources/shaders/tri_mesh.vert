#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;

layout (location = 0) out vec3 outColor;

layout (set = 0, binding = 0) uniform CameraBuffer {
	mat4 view;
	mat4 proj;
	mat4 viewProj;
} cameraData;

//! Push constant layout
layout (push_constant) uniform constants
{
	vec4 data;
	mat4 render_matrix;
} PushConstants;

void main()
{
	outColor = color;
	mat4 transform = cameraData.viewProj * PushConstants.render_matrix;
	gl_Position = transform * vec4(position, 1.0f);
}