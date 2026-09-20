import numpy as np
import OpenGL.GL as gl
from visual.shader_builder import compile_shader
from visual.objects.lines import Line


class PointPair(Line):

    """
    a CGA point pair P1^P2: its two points, and the chord joining them

    Same shader, VAO and GL_LINES draw as Line, but bounded -- a point pair is
    a finite object, so the chord is never stretched to Line.EXTENT. The dots
    come from the point shader reading the very same buffer: GL_LINES takes the
    vertices two at a time, GL_POINTS takes all of them.
    """

    def __init__(
            self,
            vertex_src:str,
            fragment_src:str,
            point_vertex_src:str,
            point_fragment_src:str
    ):
        super().__init__(vertex_src, fragment_src)
        self.dot_program = compile_shader(point_vertex_src, point_fragment_src)


    def update(self, pairs):
        # pairs: iterable of ((x1, y1), (x2, y2)), i.e. anything shaped (N, 2, 2)
        endpoints = np.asarray(pairs, dtype=np.float32).reshape(-1, 2)
        self.count = endpoints.shape[0]  # vertex count, i.e. 2 per pair

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, endpoints.nbytes, endpoints, gl.GL_DYNAMIC_DRAW)


    def draw(self,
             projection:np.ndarray,
             color=(0.2, 1.0, 0.6, 1.0),
             width=3.0,
             size=10):

        super().draw(projection, color, width)  # the chords
        if self.count == 0:
            return

        gl.glUseProgram(self.dot_program)
        loc_projection = gl.glGetUniformLocation(self.dot_program, "uProjection")
        loc_color = gl.glGetUniformLocation(self.dot_program, "uColor")
        loc_size = gl.glGetUniformLocation(self.dot_program, "uPointSize")

        gl.glUniformMatrix4fv(loc_projection, 1, gl.GL_TRUE, projection)  # gl.GL_TRUE: matrix given row-major

        gl.glUniform4f(loc_color, *color)
        gl.glUniform1f(loc_size, size)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        gl.glBindVertexArray(0)
