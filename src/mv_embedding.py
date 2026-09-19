import numpy as np
from src.CGA_2D import *


GP_TABLE = build_gp_table()
OUTER_TABLE = build_outer_table()
INNER_TABLE = build_inner_table()

inner_product = generate_operator(INNER_TABLE)
outer_product = generate_operator(OUTER_TABLE)
gp = generate_operator(GP_TABLE)


def point_to_mv(x):

    if x.ndim == 1:
        x = x[None, :]

    P = np.zeros((x.shape[0], 16))
    P += x[:, 0:1] * e1                                    # e1
    P += x[:, 1:2] * e2                                    # e2
    P += eo                                                # eo
    P += (0.5 * np.sum(x**2, axis=1, keepdims=True)) * einf  # einf
    return P

def mv_to_point(P):

    if P.ndim == 1:
        P = P[None, :]

    x = np.zeros((P.shape[0], 2))

    # normalization (inner_product returns a full multivector; the dot
    # product's actual value lives in its scalar component, index 0)
    inner = inner_product(-P, einf)[:, 0]
    P_norm = P / inner[:, None]

    # extract coordinates
    x[:, 0] = inner_product(P_norm, e1)[:, 0]
    x[:, 1] = inner_product(P_norm, e2)[:, 0]
    return x


def dual_line_mv(n, d):

    """
    build the mv of a dual line
    n = nx e1 + ny e2  {normal to the line, unitary}
    d = scalar, distance form origin
    return L = n + d * e_inf
    """
    d = np.asarray(d).reshape(-1, 1)
    mv = np.zeros((n.shape[0], 16))
    mv += n[:, 0:1] * e1
    mv += n[:, 1:2] * e2
    distance = d * einf
    L = mv + distance
    return L

def euc_line_from_dual_line_mv(mv):
    
    # recover the unit normal n = nx*e1 + ny*e2 via inner products
    nx = inner_product(mv, e1)[:, 0:1]
    ny = inner_product(mv, e2)[:, 0:1]
    normal = nx * e1 + ny * e2

    # eo . einf = -1, and e1/e2 are orthogonal to eo, so mv . eo = -d
    d = -inner_product(mv, eo)[:, 0:1]

    # rotate the normal by 90 deg (multiply by the 2D pseudoscalar e1^e2) to get the tangent direction
    e12 = gp(e1, e2)
    v_mv = gp(normal, e12)

    # closest point on the line to the origin
    x0_mv = d * normal

    # e1, e2 sit at blade indices 1, 2 -> drop straight to euclidean (N, 2) coords
    x0 = x0_mv[:, 1:3]
    v = v_mv[:, 1:3]
    return x0, v



def center_radius_from_dual_circle(C_start):
    center = np.zeros((C_start.shape[0], 2))
    inner = inner_product(-C_start, einf)[:, 0]
    C_norm = C_start / inner[:, None]
    center[:, 0] = inner_product(C_norm, e1)[:, 0]
    center[:, 1] = inner_product(C_norm, e2)[:, 0]
    squared_C = inner_product(C_norm, C_norm)[:, 0]
    rad = np.sqrt(squared_C)

    return center, rad[:, None]

def dual_circle_from_CR(center, radius):
    P = point_to_mv(center)
    P += -0.5 * (radius**2)*einf
    return P

def direct_circle_from_euc_p(p1, p2, p3):
    P1 = point_to_mv(p1)
    P2 = point_to_mv(p2)
    P3 = point_to_mv(p3)
    C = outer_product(P1, P2)
    C = outer_product(C, P3)
    return C



from src.CGA_2D_op import rotation, rotation_rev, translation, translation_rev, even_sandwhich

def motor_of(theta, t):
    R, Rr = rotation(theta), rotation_rev(theta)
    T, Tr = translation(*t), translation_rev(*t)
    return gp(T, R), gp(Rr, Tr)

def extract_pose(M, M_rev):
    e12 = gp(e1, e2)
    scalar_part = M[0]
    e12_coeff = -inner_product(M[None,:], e12[None,:])[:,0][0]
    theta = 2*np.arctan2(-e12_coeff, scalar_part)
    P0 = point_to_mv(np.array([[0.0,0.0]]))
    pos = mv_to_point(even_sandwhich(P0, M, M_rev))[0]
    return pos, theta
