import numpy as np
import OpenGL.GL as gl
from visual.shader_builder import compile_shader

SEGMENTS_PER_AXIS = 3  # shaft + two arrowhead wings


def _rotate(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c], dtype=np.float32)


def _arrow_segments(origin, direction, length, head_ratio, head_angle):
    # shaft from `origin` to the tip, plus two backward-angled wings at the
    # tip, i.e. the classic "->"-style arrowhead made only of line segments
    tip = origin + direction * length
    back = -direction
    wing_length = length * head_ratio
    wing1 = tip + _rotate(back, head_angle) * wing_length
    wing2 = tip + _rotate(back, -head_angle) * wing_length
    return [(origin, tip), (tip, wing1), (tip, wing2)]


class Frame():

    def __init__(
            self,
            vertex_src:str,
            fragment_src:str,
            axis_length=1.0,
            head_ratio=0.25,
            head_angle=np.radians(25.0)
    ):
        self.shader_program = compile_shader(vertex_src, fragment_src)
        self.axis_length = axis_length
        self.head_ratio = head_ratio
        self.head_angle = head_angle
        self.count = 0

        self.x_vao, self.x_vbo = self._make_buffer()
        self.y_vao, self.y_vbo = self._make_buffer()


    def _make_buffer(self):
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)

        # layout loc -> 0 : endpoint position
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 2 * 4, None)
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)

        return vao, vbo


    def update(self, frames):
        # frames: iterable of (x, y, theta) SE(2) poses; theta in radians,
        # measured from the world's x-axis
        frames = list(frames)
        vertices_per_axis = SEGMENTS_PER_AXIS * 2
        x_endpoints = np.empty((len(frames) * vertices_per_axis, 2), dtype=np.float32)
        y_endpoints = np.empty((len(frames) * vertices_per_axis, 2), dtype=np.float32)

        for i, (x, y, theta) in enumerate(frames):
            origin = np.array([x, y], dtype=np.float32)
            x_axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            y_axis = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)  # x_axis rotated +90deg

            x_segments = _arrow_segments(origin, x_axis, self.axis_length, self.head_ratio, self.head_angle)
            y_segments = _arrow_segments(origin, y_axis, self.axis_length, self.head_ratio, self.head_angle)

            for seg_i, (a, b) in enumerate(x_segments):
                x_endpoints[i * vertices_per_axis + 2 * seg_i] = a
                x_endpoints[i * vertices_per_axis + 2 * seg_i + 1] = b
            for seg_i, (a, b) in enumerate(y_segments):
                y_endpoints[i * vertices_per_axis + 2 * seg_i] = a
                y_endpoints[i * vertices_per_axis + 2 * seg_i + 1] = b

        self.count = x_endpoints.shape[0]  # vertex count, i.e. 6 per frame (3 segments)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.x_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, x_endpoints.nbytes, x_endpoints, gl.GL_DYNAMIC_DRAW)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.y_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, y_endpoints.nbytes, y_endpoints, gl.GL_DYNAMIC_DRAW)


    def draw(self,
             projection:np.ndarray,
             width=2.0,
             x_color=(1.0, 0.0, 0.0, 1.0),
             y_color=(0.0, 1.0, 0.0, 1.0)):

        if self.count == 0:
            return

        gl.glUseProgram(self.shader_program)
        loc_projection = gl.glGetUniformLocation(self.shader_program, "uProjection")
        loc_color = gl.glGetUniformLocation(self.shader_program, "uColor")

        gl.glUniformMatrix4fv(loc_projection, 1, gl.GL_TRUE, projection)  # gl.GL_TRUE: matrix given row-major
        gl.glLineWidth(width)

        gl.glUniform4f(loc_color, *x_color)
        gl.glBindVertexArray(self.x_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self.count)

        gl.glUniform4f(loc_color, *y_color)
        gl.glBindVertexArray(self.y_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self.count)

        gl.glBindVertexArray(0)
