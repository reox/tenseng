"""
Use the library to calculate stiffness matrices for Zysset-Curnier Model based orthotropic materials
and extract the engineering constants from the stiffness matrices.
"""
import numpy as np
import tenseng.tenseng as t
I = t.identity(2)


def C_iso(l_0, mu_0, k, rho=1):
    assert 0 <= rho <= 1
    return np.asarray(t.to_matrix(l_0 * rho**k * t.dyad(I, I) + 2 * mu_0 * rho**k * t.double_tensor_product(I, I)))


def C_zysset(l_0, lp_0, mu_0, k, l, rho=1, M=np.eye(3)):
    assert np.trace(M) == 3
    eval, evec = np.linalg.eig(M)
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


def iso_to_constants(C):
    """Return Material constants E and nu for an isotropic stiffness tensor"""
    assert C.shape == (6,6)
    # Checks for isotropy
    assert np.all(C * np.array([[0,0,0,1,1,1], [0,0,0,1,1,1], [0,0,0,1,1,1],[1,1,1,0,1,1],[1,1,1,1,0,1], [1,1,1,1,1,0]]) == 0)
    assert C[0,0] == C[1,1]
    assert C[1,1] == C[2,2]
    assert C[3,3] == C[4,4]
    assert C[4,4] == C[5,5]
    assert C[0, 1] == C[0, 2]
    assert C[1, 2] == C[0, 2]
    assert C[1, 0] == C[2, 0]
    assert C[2, 1] == C[2, 0]
    assert C[0, 1] == C[1, 0]

    G = C[3,3]
    l = C[0, 1]
    E = (G*(3*l + 2*G)) / (l + G)
    nu = l / (2 * (l + G))

    return E, nu

# Values are from Gross et al. (2013) Biomech Model Mechanobiol. DOI: 10.1007/s10237-012-0443-2
print("Isotropic Model for femur:")
print(C_iso(3586.8, 3731.1, 1.61))
for rho in np.linspace(0.01, 1, 20):
    print(rho, iso_to_constants(C_iso(3586.8, 3731.1, 1.61, rho)))

print("Zysset Model for femur:")
print(C_zysset(4609.2, 3691.6, 3738.2, 1.60, 0.99))
for rho in np.linspace(0.01, 1, 20):
    print(rho, iso_to_constants(C_zysset(4609.2, 3691.6, 3738.2, 1.60, 0.99, rho)))


print('----------------------')
for rho in np.linspace(0.01, 1, 20):
    #print(rho, iso_to_constants(C_zysset(5789.8, 4068.9, 4051.5, 1.77, 1.18, rho)))
    #print(rho, iso_to_constants(C_zysset(6878.6, 4364.2, 4155.9, 1.71, 1.12, rho)))
    #print(rho, iso_to_constants(C_zysset(4982.4, 3518.5, 3470.7, 1.62, 1.10, rho)))
    pass



