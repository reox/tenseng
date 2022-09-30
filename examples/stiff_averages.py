"""
Fun with stiffness matrices!

The plotting routine is inspired by https://github.com/coudertlab/elate
The Voigt-Reuss-Hill Averaging was directly copied from there.

Note the following relationships of stresses and strains:
sigma_ij ... stress tensor
epsilon_ij ... strain tensor

1) C ...  stiffness tensor
sigma_ij = C_ijkl epsilon_kl

2) K / S / C^-1 ... compliance tensor
epsilon_ij = K_ijkl sigma_kl


Copyright (c) 2022 S. Bachmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np


def ortho_stiffness(E_1, E_2, E_3, G_23, G_31, G_12, nu_12, nu_13, nu_23):
    """
    Returns an orthotropic stiffness matrix from 9 material constants

    Note that G_ij = G_ji but nu_ij != nu_ji"""
    S = np.zeros((6, 6))
    nu_21 = (nu_12 / E_1) * E_2
    nu_31 = (nu_13 / E_1) * E_3
    nu_32 = (nu_23 / E_2) * E_3
    S[0,0] = 1 / E_1
    S[1,1] = 1 / E_2
    S[2,2] = 1 / E_3
    S[3,3] = 1 / G_23
    S[4,4] = 1 / G_31
    S[5,5] = 1 / G_12
    S[0,1] = -nu_21 / E_2
    S[0,2] = -nu_31 / E_3
    S[1,0] = -nu_12 / E_1
    S[1,2] = -nu_32 / E_3
    S[2,0] = -nu_13 / E_1
    S[2,1] = -nu_23 / E_2
    return np.linalg.inv(S)


def iso_stiffness(E, nu):
    """Isotropic stiffness, defined by 2 material constants"""
    G = E / (2*(1+nu))
    return ortho_stiffness(E, E, E, G, G, G, nu, nu, nu)


def cubicansio_stiffness(E, G, nu):
    """Cubic Anisotropic Material"""
    # From: https://de.wikipedia.org/wiki/Orthotropie#Spezialf%C3%A4lle_der_Orthotropie
    return ortho_stiffness(E, E, E, G, G, G, nu, nu, nu)


def transiso_stiffness(E_1, E_3, nu_12, nu_31, G_23):
    """
    Transversal isotropic material, defined by 5 constants

    * E1 = E2 ... minor direction
    * E3 ... major direction
    * nu_12 = nu_21
    * nu_31 = nu_32
    * G_23 = G_31
    * G_12 = E_1 / (2*(1-nu_12)
    """
    # symmetries above orthotropic:
    E_2 = E_1
    nu_32 = nu_31
    G_31 = G_23
    G_12 = E_1 / (2*(1-nu_12))
    nu_23 = (nu_32 / E_3) * E_2
    nu_13 = (nu_31 / E_3) * E_1

    return ortho_stiffness(E_1, E_2, E_3, G_23, G_31, G_12, nu_12, nu_13, nu_23)


avg_res = namedtuple("Averages", ['Voigt', 'Reuss', 'Hill'])
consts = namedtuple('ElasticConstants', ['BulkModulus', 'YoungsModulus', 'ShearModulus', 'PoissonRatio'])


def _avg(M):
    return (M[0][0] + M[1][1] + M[2][2]) / 3, (M[1][2] + M[0][2] + M[0][1]) / 3, (M[3][3] + M[4][4] + M[5][5]) / 3


def averages(C: np.array):
    """
    Returns the Voigt, Reuss and Hill averages for a given stiffness matrix

    Voigt: Takes averages only from the stiffness matrix
    Reuss: Takes averages only from the compliance matrix
    Hill: Takes averages of Voigt and Reuss

    From: https://github.com/coudertlab/elate/blob/8c789b7f249d41219a8dbe12974820387b6d5c55/elastic.py#L689
    """
    S = np.linalg.inv(C)

    A, B, C = _avg(C)
    a, b, c = _avg(S)

    KV = (A + 2 * B) / 3
    GV = (A - B + 3 * C) / 5

    KR = 1 / (3 * a + 6 * b)
    GR = 5 / (4 * a - 4 * b + 3 * c)

    KH = (KV + KR) / 2
    GH = (GV + GR) / 2

    return avg_res(
        consts(KV, 1 / (1 / (3 * GV) + 1 / (9 * KV)), GV, (1 - 3 * GV / (3 * KV + GV)) / 2),
        consts(KR, 1 / (1 / (3 * GR) + 1 / (9 * KR)), GR, (1 - 3 * GR / (3 * KR + GR)) / 2),
        consts(KH, 1 / (1 / (3 * GH) + 1 / (9 * KH)), GH, (1 - 3 * GH / (3 * KH + GH)) / 2),
    )


def _print(M):
    for row in M:
        for val in row:
            print(f'{val: .6f}', end=" ")
        print("")


def _C_to_S_mat(C):
    """Convert a stiffness matrix into a compliance tensor

    from: https://github.com/coudertlab/elate/blob/8c789b7f249d41219a8dbe12974820387b6d5c55/elastic.py#L624
    """
    assert C.shape == (6, 6)
    S = np.linalg.inv(C)
    voigt_positions = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
    def coeff(p, q): return 1. / ((1 + p // 3) * (1 + q // 3))

    return [
        [
            [
                [
                    coeff(voigt_positions[i][j], voigt_positions[k][l]) * S[voigt_positions[i][j]][voigt_positions[k][l]]
                    for i in range(3)
                ]
                for j in range(3)
            ]
            for k in range(3)
        ]
        for l in range(3)
    ]


def plot_elastic_3d(C, npoints=100, cmap='jet', title=None):
    """
    Plot the Young's Modulus of a stiffness matrix in 3D
    :param C: the stiffness matrix
    :param npoints: number of sample points on the surface in u and v direction (surface will have npoints**2 points)
    :param cmap: name of the colormap to use
    """
    S_mat = _C_to_S_mat(C)

    U, V = np.meshgrid(np.linspace(0, np.pi, npoints), np.linspace(0, 2 * np.pi, npoints))
    A = [np.sin(U) * np.cos(V), np.sin(U) * np.sin(V), np.cos(U)]

    # get the Youngs Modulus in the direction (u, v)
    R = 1 / np.array([A[i] * A[j] * A[k] * A[l] * S_mat[i][j][k][l]
        for i in range(3) for j in range(3) for k in range(3) for l in range(3)
    ]).sum(axis=0)

    X, Y, Z = R * A

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Color the faces according to the E-modulus from min to max:
    min_R, max_R = np.min(R), np.max(R)
    # If the material is isotropic, we need to handle the colorbar differently...
    if np.isclose(max_R - min_R, 0):
        facecolor = plt.cm.get_cmap(cmap)(np.zeros(R.shape) + 0.5)
        normalize = plt.cm.colors.Normalize(min_R - 0.5, max_R + 0.5)
    else:
        facecolor = plt.cm.get_cmap(cmap)((R - min_R) / (max_R - min_R))
        normalize = plt.cm.colors.Normalize(min_R, max_R)

    ax.plot_surface(X, Y, Z, rcount=npoints, ccount=npoints, facecolors=facecolor)
    ax.set_xlim(-max_R * 1.05, max_R * 1.05)
    ax.set_ylim(-max_R * 1.05, max_R * 1.05)
    ax.set_zlim(-max_R * 1.05, max_R * 1.05)
    # Generate the colorbar on our own...
    plt.colorbar(plt.cm.ScalarMappable(normalize, plt.cm.get_cmap(cmap)), label='Youngs Modulus in GPa')
    if title:
        plt.title(title)
    plt.show()


if __name__ == "__main__":
    # Force matplotlib to use interactive windows. Might not be required in all cases...
    #mpl.use('Qt5Agg')

    table_3 = {
        # Content of Table 3 in Book: Bone Mechanics (1989) by Stephen C. Cowin, p.102
        # Gives the elastic constants for Human cortical bone
        # Values in GPa
        'Reilly and Burstein': transiso_stiffness(11.5, 17.0, 0.58, 0.46, 3.3),  # Standard machine testing
        'Yoon and Katz': transiso_stiffness(18.8, 27.4, 0.312, 0.281, 8.71),  # Ultrasonic measurement
        'Knets et al.': ortho_stiffness(6.91, 8.51, 18.4, 4.91, 3.56, 2.41, 0.49, 0.12, 0.14),  # machine
        'Ashman et al.': ortho_stiffness(12.0, 13.4, 20.0, 6.23, 5.61, 4.53, 0.376, 0.222, 0.235),  # ultrasonic

        # Some more values for fun and profit, not contained in that table:
        'Isotropic': iso_stiffness(12, 0.3),
        # A stiffness tensor from a homogenization (values in MPa)
        'Anisotropic': np.array([
            [505.27, 102.183, 215.497, 2.6955, 0.755812, 1.49247],
            [102.183, 183.834, 118.513, 2.13102, -5.87697, 8.72265],
            [215.497, 118.513, 939.81, -2.09258, 0.313404, -4.04018],
            [2.6955, 2.13102, -2.09258, 112.866, -4.78108, -4.79793],
            [0.755812, -5.87697, 0.313404, -4.78108, 257.989, 3.6341],
            [1.49247, 8.72265, -4.04018, -4.79793, 3.6341, 84.2374],
        ]) / 1000,
        # The same stiffness after orthotropization:
        'orthotropized': ortho_stiffness(0.42243, 0.15542, 0.81241, 0.11245, 0.25739, 0.083438, 0.44541, 0.017308, 0.089041),
    }

    for k, C in table_3.items():
        print(k)
        # Use this as an input to http://progs.coudert.name/elate
        _print(C)

        plot_elastic_3d(C, title=k)
        for name, res in averages(C)._asdict().items():
            print(f' {name:10s}', res)
        print("------------------------------")
