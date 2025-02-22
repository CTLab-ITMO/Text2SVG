# src/geometry.py

class Circle:
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center


class Ellipse:
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center


class Path:
    def __init__(self, num_control_points, points, is_closed):
        self.num_control_points = num_control_points
        self.points = points
        self.is_closed = is_closed


class Polygon:
    def __init__(self, points, is_closed):
        self.points = points
        self.is_closed = is_closed


class Rect:
    def __init__(self, p_min, p_max):
        self.p_min = p_min
        self.p_max = p_max
