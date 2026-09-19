from src.CGA_2D import *

GP_TABLE = build_gp_table()
gp = generate_operator(GP_TABLE)

pseudoscalar = basis(15) # e1^e2^ep^en
e12 = gp(e1, e2) 
E = gp(en, ep)

def dual(mv):
    return gp(-pseudoscalar, mv)

def rotation(theta):
    return np.cos(theta/2) * scalar - np.sin(theta/2)*e12

def dilatation(ln_lambda):
    return np.cosh(ln_lambda/2) * scalar + np.sinh(ln_lambda/2) * E

def translation(tx, ty):
    t = tx*e1 + ty*e2
    return scalar - 0.5*gp(t, einf)

def inversion(cx=0.0, cy=0.0, radius=1.0):
    # dual circle at (cx, cy) with given radius; use with odd_sandwhich to invert
    point = cx*e1 + cy*e2 + eo + 0.5*(cx**2 + cy**2)*einf
    return point - 0.5*(radius**2)*einf

def rotation_rev(theta):
    return np.cos(theta/2) * scalar + np.sin(theta/2)*e12

def dilatation_rev(ln_lambda):
    return np.cosh(ln_lambda/2) * scalar - np.sinh(ln_lambda/2) * E

def translation_rev(tx, ty):
    t = tx*e1 + ty*e2
    return scalar + 0.5*gp(t, einf)

def inversion_rev(cx=0.0, cy=0.0, radius=1.0):
    # s is a vector (s~ = s) with s^2 = radius^2, so s^-1 = s / radius^2
    return inversion(cx, cy, radius) / (radius**2)



def even_sandwhich(mv, rotor, rotor_rev):
    return gp(gp(rotor, mv), rotor_rev)

def odd_sandwhich(mv, rotor, rotor_rev):
    return -gp(gp(rotor, mv), rotor_rev)