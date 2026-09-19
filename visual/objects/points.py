import numpy as np
import OpenGL.GL as gl
from visual.shader_builder import compile_shader

class Point():

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
        # layout loc -> 0 : position
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 2 * 4, None)
        gl.glEnableVertexAttribArray(0)


        gl.glBindVertexArray(0)


    def update(self, points):
        data = np.asarray(points, dtype=np.float32).flatten()
        self.count = len(points) 

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)


    def draw(self, 
             projection:np.ndarray,
             color=(1.0, 1.0, 1.0, 1.0), 
             size=10):

        if self.count == 0:
            return

        
        gl.glUseProgram(self.shader_program)
        loc_projection = gl.glGetUniformLocation(self.shader_program, "uProjection")
        loc_color = gl.glGetUniformLocation(self.shader_program, "uColor")
        loc_size = gl.glGetUniformLocation(self.shader_program, "uPointSize")

        gl.glUniformMatrix4fv(loc_projection, 1, gl.GL_TRUE, projection)  # gl.GL_TRUE: matrix given row-major

        gl.glUniform4f(loc_color, *color)
        gl.glUniform1f(loc_size, size)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        gl.glBindVertexArray(0)

