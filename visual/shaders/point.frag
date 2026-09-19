// point.frag
#version 330 core
out vec4 FragColor;

uniform vec4 uColor;

void main()
{
    // gl_PointCoord goes 0..1 across the point sprite; remap to -1..1
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (length(coord) > 1.0)
        discard; // cut the square into a circle
    FragColor = uColor;
}