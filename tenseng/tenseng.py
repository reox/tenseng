from itertools import product

class Vector(object):
    """
    Vector Object for tensors of at least 1. rank
    Scalar values (tensors 0. rank) can be used as scalars (e.g. float/int)

    Tensors of other dimensions of 3 might cause problems ;)
    """
    def __init__(self, x = 0, y = 0, z = 0, dim = 3):
        self.dim = dim
        self.data = [x, y, z]

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError("indices must be int")
        if not 0 <= key < self.dim:
            raise IndexError("indices must be between 0 and {}".format(self.dim - 1))

        return self.data[key]

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError("indices must be int")
        if not 0 <= key < self.dim:
            raise IndexError("indices must be between 0 and {}".format(self.dim - 1))
        # Is there no better way for those checks?
        if isinstance(self.data[key], (int, float)) and not isinstance(value, (int, float)):
            raise ValueError("Can not set non-scalar on scalar data")
        elif isinstance(value, (int, float)) and not isinstance(self.data[key], (int, float)):
            raise ValueError("Can not set scalar on non-scalar data")
        elif not (isinstance(value, (int, float)) and isinstance(self.data[key], (int, float))) and self.data[key].rank() != value.rank():
            raise ValueError("Can not set non-equal rank items")

        self.data[key] = value

    def __add__(self, other):
        if self.rank() != other.rank():
            raise ValueError("not possible to add different ranks")
        res = null(self.rank())
        for i in range(self.dim):
            res[i] = self[i] + other[i]

        return res

    def __sub__(self, other):
        if self.rank() != other.rank():
            raise ValueError("not possible to subtract different ranks")
        res = null(self.rank())
        for i in range(self.dim):
            res[i] = self[i] - other[i]

        return res

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            # Multiply by scalar
            # This is defined for all ranks in the same way.
            res = null(self.rank())
            for i in range(self.dim):
                res[i] = self[i] * other
            return res
        else:
            # Now it gets tricky...
            # We have several things here:
            # a_i * b_i = c         Normal dot product for two vectors
            # A_ik * B_kj = C_ij    dot product of two 2nd r. tensors
            # A_ij * u_j = c_i      dot product of vector and 2nd r. tensor

            if self.rank() == 1 and other.rank() == 1:
                # This is the normal scalar product
                res = 0
                for i in range(self.dim):
                    res += self[i] * other[i]
                return res
            elif self.rank() == 2 and other.rank() == 2:
                # dot product for tensors
                res = null(self.rank())
                for i, j, k in product(range(self.dim), repeat=3):
                    res[i][j] += self[i][k] * other[k][j]
                return res
            elif self.rank() == 2 and other.rank() == 1:
                # u * A is not defined...
                res = null(other.rank())
                for i, j in product(range(self.dim), repeat=2):
                    res[i] += self[i][j] * other[j]
            else:
                raise ValueError("dot product is not defined for this type of eq.")

    # Define the other side too (needed for multiplication with scalar)
    def __rmul__(self, other):
        # The only case where this is needed is to multiply scalars
        if isinstance(other, (float, int)):
            # Multiply by scalar
            # This is defined for all ranks in the same way.
            res = null(self.rank())
            for i in range(self.dim):
                res[i] = self[i] * other
            return res
        else:
            raise ValueError("this type of multiplication is not defined")

    def __matmul__(self, other):
        # cross product for 1. rank tensors
        if self.rank() != other.rank():
            raise ValueError("not possible create cross product with different ranks")
        if self.rank() != 1:
            raise ValueError("cross product is defined for first order tensors only")
        res = Vector(dim=self.dim)
        for i, j, k in product(range(self.dim), repeat=3):
            res[i] += levi[i][j][k] * self[j] * other[k]

        return res

    def __str__(self):
        return "[{}, {}, {}]".format(self[0], self[1], self[2])

    def __repr__(self):
        return "<{} rank Tensor of {}>".format(self.rank(), self)

    def rank(self):
        # return the rank of the tensorial object
        if isinstance(self[0], (float, int)):
            return 1
        return self[0].rank() + 1

    def get(self, *args):
        # A wrapper function for variable number of arguments
        elem = self
        for x in args:
            elem = elem[x]
        return elem

    def set(self, *args, value):
        # A wrapper function for variable number of arguments
        # Find the parent element
        elem = self.get(*args[:-1])
        # Set the value
        elem[args[-1]] = value

    def dof(self):
        # Return the number of Degrees of Freedom
        return self.dim ** self.rank()

    def as_list(self):
        x = []
        for i in range(self.dim):
            if not isinstance(self[i], (int, float)):
                x.append(self[i].as_list())
            else:
                x.append(self[i])
        return x


# Kroneker Delta
kron = Vector(Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1))

# Levi Civita
levi = Vector(Vector(Vector(0, 0, 0), Vector(0, 0, 1), Vector(0, -1, 0)),
              Vector(Vector(0, 0, -1), Vector(0, 0, 0), Vector(1, 0, 0)),
              Vector(Vector(0, 1, 0), Vector(-1, 0, 0), Vector(0, 0, 0)))


def null(rank):
    # Returns a null tensor for the given rank
    if rank == 0:
        return 0
    return Vector(null(rank-1), null(rank-1), null(rank-1))


def identity(rank):
    # Return the identity tensor for a given rank
    # I * v = v * I = v
    # A = I : A
    # TODO there is also A^T = \bar{I} : A...
    # We can define this via a orthonormal basis and dyads.

    # We do not have an identity for ranks 0,1,3 and above 4...
    if rank not in (2, 4):
        raise ValueError("Identity is not defined for rank {}".format(rank))
    x = null(rank)
    # For rank == 2, we set all fields to 1 where i = j
    # For rank == 4, we set all fields to 1 where i = k and j = l
    if rank == 2:
        for i in range(3):
            x.set(i, i, value=1)
    if rank == 4:
        for i, k in product(range(3), repeat=2):
            x.set(i, k, i, k, value=1)
    return x



def dyad(a, b):
    if a.rank() != b.rank():
        raise ValueError("Dyad is not defined for different ranks")
    nrank = a.rank() + b.rank()
    res = null(nrank)
    for t in product(range(3), repeat=nrank):
        res.set(*t, value=(a.get(*t[:nrank//2]) * b.get(*t[nrank//2:])))
    return res


def double_tensor_product(a, b):
    """
    Calculate the double tensorial product (X), defined by Curnier et al. (1994):
    A (X) B = 0.5 * (A_ik B_jl + A_il B_jk) e_i (x) e_j (x) e_k (x) e_l
    where (x) is the dyadic product and e_i are the basis vectors.
    """
    if a.rank() != 2 or b.rank() != 2:
        raise ValueError("Two tensors of rank 2 has to be supplied")
    res = null(4)  # Result is of order 4.
    for i, j, k, l in product(range(3), repeat=4):
        # TODO: What is with the basis? here, we skip it, because we usually have standard basis anyways
        res.set(i, j, k, l, value=0.5 * (a[i][k] * b[j][l] + a[i][l] * b[j][k]))
    return res


def double_contraction(a, b):
    # In the double contraction, the rank of b must be less or equal than of a
    if b.rank() > a.rank():
        raise ValueError("The operation is not defined for lower rank tensor b")

    # Basically it works like this: a_i..z : b_w..z = c_i..v
    # Which means, the resulting vector has mute indices of a
    # and b uses all indices as dummy (and those are the n last of a,
    # when n is the rank of b)
    d = a.rank() - b.rank()
    res = null(d)
    for t in product(range(3), repeat=a.rank()):
        if d == 0:
            res += a.get(*t) * b.get(*t[d:])
        else:
            res.set(*t[:d], value=(a.get(*t) * b.get(*t[d:])) + res.get(*t[:d]))
    return res


def trace(a):
    res = 0
    for i in range(3):
        res += a[i][i]
    return res


# TODO need to define for higher orders... A_ijkl => A_klij
def transpose(a):
    res = null(a.rank())
    for i, j in product(range(3), repeat=2):
        res[i][j] = a[j][i]
    return res


def is_sym(a):
    """returns true if tensor is symmetric"""
    return a == transpose(a)


def sym(a):
    # Symmetric tensor part
    # 1/2 * (a + aT)
    a = 0.5 * (a + transpose(a))
    return a


def skew(a):
    # Antisymmetric tensor part
    # 1/2 * (a - aT)
    a = 0.5 * (a - transpose(a))
    return a


def vol(a):
    # Volumetric (spherical) part of 2nd rank tensor
    return 1.0/3.0 * (trace(a) * identity(a.rank()))


def dev(a):
    # Deviatoric part of 2nd rank tensor
    return a - 1.0/3.0 * (trace(a) * identity(a.rank()))


def det(a):
    # Calculate the determinant of a second rank tensor
    res = 0
    for i, j, k in product(range(3), repeat=3):
        res += levi[i][j][k] * a[i][0] * a[j][1] * a[k][2]
    return res


def to_matrix(c):
    """return the Voigt notation of a 4th order tensor"""
    if c.rank() != 4:
        raise ValueError("Can give matrix notation only for 4th order tensor")

    # Voigt Order
    Z = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    for i, j, k, l in product(range(3), repeat=4):
        # To write a tensor in Voigt notation, we need to make sure the minor symmetries are fulfilled.
        if c[i][j][k][l] != c[j][i][k][l]:
            raise ValueError(f"Minor Symmetry in first index pair not fulfilled at {(i, j, k, l)}")
        if c[i][j][k][l] != c[i][j][l][k]:
            raise ValueError(f"Minor Symmetry in second index pair not fulfilled at {(i, j, k, l)}")

    return [[c[i][j][k][l] for k, l in Z] for i, j in Z]


