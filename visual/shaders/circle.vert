// circle.vert
#version 330 core
layout (location = 0) in vec2 aCenter;
layout (location = 1) in float aRadius;
layout (location = 2) in vec2 aCorner; // local quad corner, -1..1, expands the circle into a billboard

uniform mat4 uProjection;

out vec2 vCorner;

void main()
{
    vec2 worldPos = aCenter + aCorner * aRadius; // quad sized directly in world units, no point-sprite size cap
    gl_Position = uProjection * vec4(worldPos, 0.0, 1.0);
    vCorner = aCorner;
}
