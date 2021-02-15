#version 450 core

layout (location = 0) out vec3 outColor;

void main()
{
	const vec3 positions[] = {
		vec3(1.0f, 1.0f, 0.0f),
		vec3(-1.0f, 1.0f, 0.0f),
		vec3(0.0f, -1.0f, 0.0f)
	};

	//const array of colors for the triangle
	const vec3 colors[3] = vec3[3](
		vec3(1.0f, 0.0f, 0.0f), //red
		vec3(0.0f, 1.0f, 0.0f), //green
		vec3(00.f, 0.0f, 1.0f)  //blue
	);

	outColor = colors[gl_VertexIndex];
	gl_Position = vec4(positions[gl_VertexIndex], 1.0f);
}