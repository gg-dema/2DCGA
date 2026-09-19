import numpy as np
import OpenGL.GL as gl
from visual.shader_builder import compile_shader

class Line():

    # half-length (world units) each line is stretched to; must exceed the
    # projection's visible extent so the segment looks like an infinite line
    EXTENT = 1e4

    def __init__(
            self,
            vertex_src:str,
            fragment_src:str
    ):
        self.shader_program = compile_shader(vertex_src, fragment_src)
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.count = 0


        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        # layout loc -> 0 : endpoint position
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 2 * 4, None)
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)


    def update(self, lines):
        # lines: iterable of (point, direction) pairs, each defining an
        # infinite line through `point` along `direction`
        endpoints = np.empty((len(lines) * 2, 2), dtype=np.float32)
        for i, (point, direction) in enumerate(lines):
            point = np.asarray(point, dtype=np.float32)
            direction = np.asarray(direction, dtype=np.float32)
            direction /= np.linalg.norm(direction)
            endpoints[2 * i] = point - direction * self.EXTENT
            endpoints[2 * i + 1] = point + direction * self.EXTENT

        self.count = endpoints.shape[0]  # vertex count, i.e. 2 per line

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, endpoints.nbytes, endpoints, gl.GL_DYNAMIC_DRAW)


    def draw(self,
             projection:np.ndarray,
             color=(1.0, 1.0, 0.0, 1.0),
             width=2.0):

        if self.count == 0:
            return

        gl.glUseProgram(self.shader_program)
        loc_projection = gl.glGetUniformLocation(self.shader_program, "uProjection")
        loc_color = gl.glGetUniformLocation(self.shader_program, "uColor")

        gl.glUniformMatrix4fv(loc_projection, 1, gl.GL_TRUE, projection)  # gl.GL_TRUE: matrix given row-major
        gl.glUniform4f(loc_color, *color)

        gl.glLineWidth(width)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self.count)
        gl.glBindVertexArray(0)
