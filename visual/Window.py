import glfw
from OpenGL.GL import *
import ctypes
import math
import random

from visual.objects.points import Point
from visual.objects.circles import Circle
from visual.objects.lines import Line
from visual.objects.frames import Frame
from visual.utils import ortho, fit_bounds, grid_lines, axis_lines, nice_step

# object sizes (point/circle radius, line width) are in screen pixels, not
# world units, so they don't shrink together with the grid below this size
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600

# world units visible top-to-bottom at MIN_WINDOW_HEIGHT; this fixes the
# world-to-pixel scale (see PIXELS_PER_UNIT), so a bigger window just shows
# more of the grid (wider field of view) instead of bigger grid squares
BASE_WORLD_HEIGHT = 10.0
PIXELS_PER_UNIT = MIN_WINDOW_HEIGHT / BASE_WORLD_HEIGHT

# zoom multiplies PIXELS_PER_UNIT. The limits are wide but finite so a stray
# flick of the wheel can't strand the view somewhere it takes fifty notches to
# come back from
MIN_ZOOM, MAX_ZOOM = 0.05, 200.0
ZOOM_PER_NOTCH = 1.15   # per scroll-wheel notch
ZOOM_PER_KEY = 1.30     # per +/- press, coarser since a key can't be flicked

# gridlines are spaced to land near this on screen at the current zoom
GRID_TARGET_SPACING_PX = 60.0


main_shaders_path = {
        "points_vertex": "visual/shaders/point.vert",
        "points_fragment": "visual/shaders/point.frag",

        "circles_vertex": "visual/shaders/circle.vert",
        "circles_fragment": "visual/shaders/circle.frag",

        "lines_vertex": "visual/shaders/line.vert",
        "lines_fragment": "visual/shaders/line.frag"    
    }

def read_shader_source(file_path):
    with open(file_path, 'r') as f:
        return f.read()

class Window:

    def __init__(self, 
                 shader_paths_dict=None,
                 width=MIN_WINDOW_WIDTH, 
                 height=MIN_WINDOW_HEIGHT, 
                 title="2DCGA"):

        self.shader_paths_dict = shader_paths_dict or main_shaders_path
        self.width = width
        self.height = height
        self.title = title

        self.window = None
        # init glfw
        if not glfw.init():
            return 
        
        # set OpenGL 3.3 Core Profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
        # create window
        self.window = glfw.create_window(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT, "2D OpenGL Canvas", None, None)
        if not self.window:
            glfw.terminate()
            return

        # can grow (grid scales fine), but not shrink past the size object sizes were tuned for
        glfw.set_window_size_limits(self.window, MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT, glfw.DONT_CARE, glfw.DONT_CARE)

        glfw.make_context_current(self.window)

        # # Enable point size configuration (so GL_POINTS can be larger than 1 pixel)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glPointSize(8.0)
        glLineWidth(2.0)

        # let faint (low-alpha) draws, like the grid, blend into the background
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.background_renderers = {
            "grid":  Line(
                vertex_src=read_shader_source(self.shader_paths_dict["lines_vertex"]),
                fragment_src=read_shader_source(self.shader_paths_dict["lines_fragment"])
            ),
            "main_axes": Line(
                vertex_src=read_shader_source(self.shader_paths_dict["lines_vertex"]),
                fragment_src=read_shader_source(self.shader_paths_dict["lines_fragment"])
            )
        }
        self.background_renderers['main_axes'].update(axis_lines())

        # View state: the world point held at the centre of the window, and the
        # factor on PIXELS_PER_UNIT. The grid is rebuilt from these rather than
        # generated once far out in every direction, so it only ever holds the
        # lines that are actually on screen however far the view zooms out.
        self.center = [0.0, 0.0]
        self.zoom = 1.0
        self.fb_size = glfw.get_framebuffer_size(self.window)
        self.projection = None
        self._drag_origin = None   # world point grabbed by a panning drag


        self.dyn_renderers = {
            "lines" :  Line(
                vertex_src=read_shader_source(self.shader_paths_dict["lines_vertex"]),
                fragment_src=read_shader_source(self.shader_paths_dict["lines_fragment"])
            ),
            "points" : Point(
                vertex_src=read_shader_source(self.shader_paths_dict["points_vertex"]),  
                fragment_src=read_shader_source(self.shader_paths_dict["points_fragment"])
            ),
            "circles" : Circle(
                vertex_src=read_shader_source(self.shader_paths_dict["circles_vertex"]),
                fragment_src=read_shader_source(self.shader_paths_dict["circles_fragment"]),
                pixels_per_unit=PIXELS_PER_UNIT
            ),
            "frames" : Frame(
                vertex_src=read_shader_source(self.shader_paths_dict["lines_vertex"]),
                fragment_src=read_shader_source(self.shader_paths_dict["lines_fragment"]),
                axis_length=1.0
            )
        }
        glfw.set_framebuffer_size_callback(
            self.window, lambda _win, width, height: self._on_resize(width, height))
        glfw.set_scroll_callback(
            self.window, lambda _win, _dx, dy: self._on_scroll(dy))
        glfw.set_mouse_button_callback(
            self.window, lambda _win, button, action, _mods: self._on_mouse_button(button, action))
        glfw.set_cursor_pos_callback(
            self.window, lambda _win, sx, sy: self._on_cursor_pos(sx, sy))
        glfw.set_key_callback(
            self.window, lambda _win, key, _scan, action, _mods: self._on_key(key, action))
        self._on_resize(*self.fb_size)



    # ---- view control: scroll or +/- to zoom, drag to pan, 0 to reset ----

    def _apply_view(self):
        # rebuild the projection, and with it the grid, from centre and zoom
        width, height = self.fb_size
        if width == 0 or height == 0:
            return
        bounds = fit_bounds(width, height, PIXELS_PER_UNIT, self.center, self.zoom)
        self.projection = ortho(*bounds)
        step = nice_step(1.0 / (PIXELS_PER_UNIT * self.zoom), GRID_TARGET_SPACING_PX)
        self.background_renderers['grid'].update(grid_lines(*bounds, step=step))

    def screen_to_world(self, sx, sy):
        # cursor positions arrive in window coordinates with y pointing down,
        # while the projection is built from framebuffer pixels; the two differ
        # on a HiDPI display, so convert through their ratio instead of assuming
        # one pixel is the other
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = self.fb_size
        dx = sx * fb_w / max(win_w, 1) - fb_w / 2.0
        dy = fb_h / 2.0 - sy * fb_h / max(win_h, 1)
        scale = PIXELS_PER_UNIT * self.zoom
        return self.center[0] + dx / scale, self.center[1] + dy / scale

    def zoom_by(self, factor, screen_pos=None):
        """Scale the zoom, holding the world point under `screen_pos` in place.

        Zooming about the cursor rather than the origin is what makes the wheel
        usable for inspecting something off-centre: the alternative magnifies
        the middle of the window and slides whatever you were looking at off
        the edge.
        """
        zoom = min(max(self.zoom * factor, MIN_ZOOM), MAX_ZOOM)
        if zoom == self.zoom:
            return
        if screen_pos is not None:
            wx, wy = self.screen_to_world(*screen_pos)
            # the grabbed point sits (w - centre) from the centre now and would
            # sit (w - centre) * zoom/zoom_new away after; absorb the difference
            k = 1.0 - self.zoom / zoom
            self.center[0] += (wx - self.center[0]) * k
            self.center[1] += (wy - self.center[1]) * k
        self.zoom = zoom
        self._apply_view()

    def reset_view(self):
        self.center, self.zoom = [0.0, 0.0], 1.0
        self._apply_view()

    def _on_resize(self, width, height):
        if width == 0 or height == 0:
            return
        self.fb_size = (width, height)
        glViewport(0, 0, width, height)
        self._apply_view()

    def _on_scroll(self, dy):
        # dy is fractional on a trackpad, so raise the per-notch factor to it
        # rather than branching on the sign
        self.zoom_by(ZOOM_PER_NOTCH ** dy, glfw.get_cursor_pos(self.window))

    def _on_mouse_button(self, button, action):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        self._drag_origin = (self.screen_to_world(*glfw.get_cursor_pos(self.window))
                             if action == glfw.PRESS else None)

    def _on_cursor_pos(self, sx, sy):
        if self._drag_origin is None:
            return
        wx, wy = self.screen_to_world(sx, sy)
        # drag the world along under the cursor: move the centre so the point
        # grabbed on press lands back under the pointer, which also leaves
        # _drag_origin correct for the next motion event
        self.center[0] += self._drag_origin[0] - wx
        self.center[1] += self._drag_origin[1] - wy
        self._apply_view()

    def _on_key(self, key, action):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.zoom_by(ZOOM_PER_KEY)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.zoom_by(1.0 / ZOOM_PER_KEY)
        elif key in (glfw.KEY_0, glfw.KEY_KP_0):
            self.reset_view()

    def step(self):
        glfw.poll_events()

        # Clear background (Dark gray)
        glClearColor(0.1, 0.1, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        self.background_renderers['grid'].draw(
            self.projection, color=(1.0, 1.0, 1.0, 0.12), width=1.0)
        self.background_renderers['main_axes'].draw(
            self.projection, color=(1.0, 1.0, 1.0, 0.5), width=2.0)

        for renderer in self.dyn_renderers.values():
            renderer.draw(self.projection)

        glfw.swap_buffers(self.window)

    def static_run(self):
        while not glfw.window_should_close(self.window):
            self.step()
        glfw.terminate()

    def is_running(self):
        return not glfw.window_should_close(self.window)

if __name__ == "__main__":

    import numpy as np
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    print(f"")

    window = Window(
        shader_paths_dict=main_shaders_path,
        width=MIN_WINDOW_WIDTH,
        height=MIN_WINDOW_HEIGHT,
        title="2DCGA")

    window.dyn_renderers['points'].update([
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0)
    ])
    window.dyn_renderers['circles'].update([
        (-2.5, -2.5, 0.3),
        (2, -2.5, 0.2),
        (2.5, 2, 0.5),
        (-2.5, 2.5, 0.3)
    ])
    window.dyn_renderers['lines'].update([
        ((-2.5, -2.5), (1.0, 0.0)),
        ((2.5, -2.5), (0.0, 1.0)),
        ((2.5, 2.5), (-1.0, 0.0)),
        ((-2.5, 2.5), (0.0, -1.0))
    ])
    window.dyn_renderers['frames'].update([
        (0.0, 0.0, 0.0), 
        (1.0, 1.0, math.pi/4),
        (-1.0, -1.0, math.pi/2),
        (2.0, -1.0, math.pi/3),
    ])
    angles = np.array([0.0, math.pi/4, math.pi/2, math.pi/3])
    steps = np.ones_like(angles) * 0.01
    while not glfw.window_should_close(window.window):

        window.dyn_renderers['frames'].update([
            (0.0, 0.0, angles[0]), 
            (1.0, 1.0, angles[1]),
            (-1.0, -1.0, angles[2]),
            (2.0, -1.0, angles[3]),
        ])
        angles += steps
        angles = np.mod(angles, 2.0 * math.pi)
        window.step()
        