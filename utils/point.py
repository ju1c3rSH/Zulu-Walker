class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def distance(self):
        import math
        return math.sqrt(self.x ** 2 + self.y ** 2)

def get_angle(p1, p2, p3):
    """
    计算角 p1-p2-p3 的角度（度数）

    Args:
        p1, p2, p3: 可以是 Point 对象或 [x, y] 数组
    """
    import math

    # 支持 Point 对象和 numpy 数组
    def get_coords(p):
        if hasattr(p, 'x'):
            return p.x, p.y
        return p[0], p[1]

    x1, y1 = get_coords(p1)
    x2, y2 = get_coords(p2)
    x3, y3 = get_coords(p3)

    # Calculate vectors
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    # Calculate dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # Calculate magnitudes
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    # Calculate angle in radians
    if mag1 == 0 or mag2 == 0:
        return 0
    angle_rad = math.acos(max(-1, min(1, dot_product / (mag1 * mag2))))

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def get_cos_angle(p1, p2, p3):
    """
    计算角 p1-p2-p3 的余弦值

    Args:
        p1, p2, p3: 可以是 Point 对象或 [x, y] 数组
    """
    import math

    # 支持 Point 对象和 numpy 数组
    def get_coords(p):
        if hasattr(p, 'x'):
            return p.x, p.y
        return p[0], p[1]

    x1, y1 = get_coords(p1)
    x2, y2 = get_coords(p2)
    x3, y3 = get_coords(p3)

    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_val = dot_product / (mag1 * mag2)
    return max(-1.0, min(1.0, cos_val))