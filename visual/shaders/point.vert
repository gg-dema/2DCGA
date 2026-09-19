// point.vert
#version 330 core
layout (location = 0) in vec2 aPos;

uniform mat4 uProjection;
uniform float uPointSize;


void main()
{
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
    gl_PointSize = uPointSize;
}