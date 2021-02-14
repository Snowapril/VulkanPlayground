#version 450 core

void main()
{
	const vec3 positions[] = {
		vec3(1.0f, 1.0f, 0.0f),
		vec3(-1.0f, 1.0f, 0.0f),
		vec3(0.0f, -1.0f, 0.0f)
	};

	gl_Position = vec4(positions[gl_VertexIndex], 1.0f);
}