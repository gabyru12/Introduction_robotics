import numpy as np

def line_points(x0, y0, x1, y1, n=100):
    x = np.linspace(x0, x1, n)
    y = np.linspace(y0, y1, n)
    return np.column_stack((x, y))


def arc_points(cx, cy, radius, start_angle, end_angle, n=100):
    theta = np.linspace(start_angle, end_angle, n)

    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    return np.column_stack((x, y))


def get_centerline():
    # =====================================================
    # Segment 1 (bottom straight)
    # =====================================================
    seg1 = line_points(
        0, -25,
        30, -25,
        n=50
    )

    # =====================================================
    # Segment 2 (right semicircle)
    # center = (30, 0)
    # radius = 25
    # from bottom to top
    # =====================================================
    seg2 = arc_points(
        cx=30,
        cy=0,
        radius=25,
        start_angle=-np.pi/2,
        end_angle=np.pi/2,
        n=100
    )

    # =====================================================
    # Segment 3 (top straight)
    # =====================================================
    seg3 = line_points(
        30, 25,
        -30, 25,
        n=100
    )

    # =====================================================
    # Segment 4 (left semicircle)
    # center = (-30, 0)
    # radius = 25
    # from top to bottom
    # =====================================================
    seg4 = arc_points(
        cx=-30,
        cy=0,
        radius=25,
        start_angle=np.pi/2,
        end_angle=3*np.pi/2,
        n=100
    )

    # =====================================================
    # Segment 5 (bottom straight)
    # =====================================================
    seg5 = line_points(
        -30, -25,
        0, -25,
        n=50
    )

    # =====================================================
    # Complete centerline
    # =====================================================
    centerline = np.vstack([
        seg1[:-1],
        seg2[:-1],
        seg3[:-1],
        seg4[:-1],
        seg5
    ])

    return centerline

centerline = get_centerline()
print(centerline)