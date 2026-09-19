// circle.frag
#version 330 core
out vec4 FragColor;

uniform vec4 uColor;
uniform float uThicknessPx; // ring thickness in screen pixels

in vec2 vCorner;

void main()
{
    // vCorner is the quad-local coordinate, -1..1 across the circle's diameter
    float dist = length(vCorner);
    // fwidth(dist) is how far dist moves per screen pixel, i.e. the reciprocal
    // of this circle's on-screen radius in pixels. Converting through it keeps
    // the ring the same visual weight at any zoom without the shader needing to
    // know the world-to-pixel scale at all
    float thickness = clamp(uThicknessPx * fwidth(dist), 0.0, 1.0);
    if (dist > 1.0 || dist < 1.0 - thickness)
        discard; // keep only a ring between [1-thickness, 1]
    FragColor = uColor;
}
