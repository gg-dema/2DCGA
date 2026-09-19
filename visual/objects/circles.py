import ctypes
import numpy as np
import OpenGL.GL as gl
from visual.shader_builder import compile_shader

# local corners of the billboard quad (2 triangles) each circle is expanded into
_QUAD_CORNERS = np.array([
    (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0),
    (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
], dtype=np.float32)

class Circle():

    def __init__(
            self,
            vertex_src:str,
            fragment_src:str,
            pixels_per_unit:float=1.0   # kept for callers; no longer used, see draw
    ):
        self.pixels_per_unit = pixels_per_unit
        self.shader_program = compile_shader(vertex_src, fragment_src)
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.vertex_count = 0

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        stride = 5 * 4  # (center_x, center_y, radius, corner_x, corner_y) per vertex
        # layout loc -> 0 : center position
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(0)
        # layout loc -> 1 : radius
        gl.glVertexAttribPointer(1, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        gl.glEnableVertexAttribArray(1)
        # layout loc -> 2 : local quad corner
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(2)
        gl.glBindVertexArray(0)


    def update(self, circles):
        # circles: iterable of (x, y, radius) triples, one per circle; each is
        # expanded into a quad (6 vertices) sized by radius in world units, so
        # circle size isn't bounded by the GPU's max point-sprite size
        centers_radii = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
        n = centers_radii.shape[0]
        self.vertex_count = n * 6

        data = np.empty((n, 6, 5), dtype=np.float32)
        data[:, :, 0:3] = centers_radii[:, np.newaxis, :]
        data[:, :, 3:5] = _QUAD_CORNERS[np.newaxis, :, :]
        data = data.reshape(-1, 5)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)


    def draw(self,
             projection:np.ndarray,
             color=(1.0, 0.0, 1.0, 1.0),
             thickness_px=3.0):

        if self.vertex_count == 0:
            return

        gl.glUseProgram(self.shader_program)
        loc_projection = gl.glGetUniformLocation(self.shader_program, "uProjection")
        loc_color = gl.glGetUniformLocation(self.shader_program, "uColor")
        loc_thickness = gl.glGetUniformLocation(self.shader_program, "uThicknessPx")

        gl.glUniformMatrix4fv(loc_projection, 1, gl.GL_TRUE, projection)  # gl.GL_TRUE: matrix given row-major
        gl.glUniform4f(loc_color, *color)
        # passed straight through in pixels: the fragment shader recovers the
        # scale from its own screen-space derivatives, so the ring stays the
        # same weight as the view zooms (converting here against a fixed
        # pixels-per-unit would fatten every ring as you zoom in)
        gl.glUniform1f(loc_thickness, thickness_px)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertex_count)
        gl.glBindVertexArray(0)
