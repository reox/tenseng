"""
Use the library to calculate stiffness matrices for Zysset-Curnier Model based orthotropic materials
and extract the engineering constants from the stiffness matrices.
"""
from collections import namedtuple

import numpy as np

import tenseng.tenseng.tenseng as t

cubic = namedtuple('CubicAnisotropic', ['E', 'nu', 'G'])
iso = namedtuple('Isotropic', ['E', 'nu'])

I = t.identity(2)


def C_iso(l_0, mu_0, k, rho=1):
    """Create an isotropic powerlaw material and return the stiffness matrix"""
    if not 0 <= rho <= 1:
        raise ValueError("Invalid Density")
    return np.asarray(t.to_matrix(l_0 * rho**k * t.dyad(I, I) + 2 * mu_0 * rho**k * t.double_tensor_product(I, I)))


def C_zysset(l_0, lp_0, mu_0, k, l, rho=1, M=np.eye(3)):
    """Create Zysset-Curnier model and return stiffness matrix"""
    if not np.isclose(np.trace(M), 3):
        raise ValueError("Invalid Fabric")
    if not 0 <= rho <= 1:
        raise ValueError("Invalid Density")
    eval, evec = np.linalg.eig(M)
    print(evec)
    M = [t.dyad(t.Vector(*evec[i]), t.Vector(*evec[i])) for i in range(3)]
    res = t.null(4)
    for i in range(3):
        res += (l_0 + 2 * mu_0) * rho ** k * eval[i] ** (2 * l) * t.dyad(M[i], M[i])
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            res += lp_0 * rho**k * eval[i]**l * eval[j]**l * t.dyad(M[i], M[j])
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            res += 2 * mu_0 * rho**k * eval[i]**l * eval[j]**l * t.double_tensor_product(M[i], M[j])

    return np.asarray(t.to_matrix(res))


def cubic_to_constants(C):
    """Return Material constants E, nu and G for an cubic-anisotropic stiffness tensor"""
    if C.shape != (6,6):
        raise ValueError("Invalid Stiffness Matrix")

    # Checks for isotropy / cubic-anisotropy
    # only the following elements must be set:
    # x y y 0 0 0
    # y x y 0 0 0
    # y y x 0 0 0
    # 0 0 0 z 0 0
    # 0 0 0 0 z 0
    # 0 0 0 0 0 z
    if not np.all(np.isclose(C * np.array(
            [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1],
             [1, 1, 1, 1, 1, 0]]), 0)) or \
            not np.isclose(C[0, 0], C[1, 1]) or \
            not np.isclose(C[1, 1], C[2, 2]) or \
            not np.isclose(C[3, 3], C[4, 4]) or \
            not np.isclose(C[4, 4], C[5, 5]) or \
            not np.isclose(C[0, 1], C[0, 2]) or \
            not np.isclose(C[1, 2], C[0, 2]) or \
            not np.isclose(C[1, 0], C[2, 0]) or \
            not np.isclose(C[2, 1], C[2, 0]) or \
            not np.isclose(C[0, 1], C[1, 0]):
        raise ValueError("Matrix does not fulfill cubic-anisotropy symmetry")

    # Invert the stiffness tensor to elasticity tensor:
    E = np.linalg.inv(C)
    e = 1 / E[0,0]  # The elastic modulus can directly be read
    nu = -E[0,1] * e  # the off-diagonal only contains the -nu/E terms
    mu = 1 / E[3,3]  # The shearmodulus can directly be read

    return cubic(e, nu, mu)


def iso_to_constants(C):
    """Returns material constants E and nu for an isotropic material """
    E, nu, G = cubic_to_constants(C)
    if not np.isclose(E / (2*(1 + nu)), G):
        raise ValueError("The Material is not isotropic!")
    return iso(E, nu)


# Values are from Gross et al. (2013) Biomech Model Mechanobiol. DOI: 10.1007/s10237-012-0443-2
# Got them from Homogenization using KUBC and 12GPa nu=0.3 material
print("Isotropic Model for combined:")
C = C_iso(3429.0, 3536.0, 1.63)
print(C)
print(iso_to_constants(C))

print("Zysset-Curnier Model for combined (KUBC; 12GPa, 0.3):")
C = C_zysset(4982.4, 3518.5, 3470.7, 1.62, 1.10)
print(C)
print(cubic_to_constants(C))
print()

# Values from Panyasantisuk et al. (2015) J.Biomech.Eng. DOI: 10.1115/1.4028968
# Used PMUBC and 10GPa nu=0.3 material and compared with the paper above.
# NOTE: They scaled the material parameters from 12 to 10GPa for the comparison
# They compared to the femur set of Gross et al.
print("Zysset-Curnier for femur KUBC based scaled (10GPa, 0.3):")
print(cubic_to_constants(C_zysset(3841.04, 3076.34, 3115.17, 1.6, 0.99)))
print("Zysset-Curnier KUBC based (10GPa, 0.3):")
print(cubic_to_constants(C_zysset(3381.79, 2745.98, 2855.14, 1.55, 0.84)))
print("Zysset-Curnier KUBC based filtered:")
print(cubic_to_constants(C_zysset(3306.34, 2735.59, 2837.43, 1.55, 0.82)))
print()
print("Zysset-Curnier PMUBC based (10GPa, 0.3):")
print(cubic_to_constants(C_zysset(6250.22, 3768.00, 3446.81, 2.01, 1.20)))
print("Zysset-Curnier PMUBC based filtered:")
print(cubic_to_constants(C_zysset(5060.06, 3353.04, 3116.88, 1.91, 1.10)))



