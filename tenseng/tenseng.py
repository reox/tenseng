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
        res = Vector(dim=self.dim)
        for i in range(self.dim):
            res[i] = self[i] + other[i]

        return res

    def __sub__(self, other):
        if self.rank() != other.rank():
            raise ValueError("not possible to subtract different ranks")
        res = Vector(dim=self.dim)
        for i in range(self.dim):
            res[i] = self[i] - other[i]

        return res

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            # Multiply by scalar
            for i in range(self.dim):
                self[i] *= other
            return self

        else:
            # otherwise: scalarproduct / double / fourfold contraction
            res = 0
            for i in range(self.dim):
                res += self[i] * other[i]

            return res

    # Define the other side too (needed for multiplication with scalar)
    __rmul__ = __mul__

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
        r = 1
        if isinstance(self[0], (float, int)):
            return r
        return r + self[0].rank()

    def get(self, *args):
        elem = self
        for x in args:
            elem = elem[x]
        return elem

    def set(self, *args, value):
        # Find the parent element
        elem = self.get(*args[:-1])
        # Set the value
        elem[args[-1]] = value

    def dof(self):
        # Return the number of Degrees of Freedom
        return self.dim ** self.rank()


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



# TODO need to define for higher orders as well
def dyad(a, b):
    if a.rank() != b.rank():
        raise ValueError("Dyad is not defined for different ranks")
    res = null(a.rank() + 1)
    for i, j in product(range(3), repeat=2):
        res[i][j] = a[i] * b[j]
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


def sym(a):
    # 1/2 * (a + aT)
    a = 0.5 * (a + transpose(a))
    return a


def skew(a):
    # 1/2 * (a - aT)
    a = 0.5 * (a - transpose(a))
    return a


def vol(a):
    pass


def dev(a):
    pass

def det(a):
    # Calculate the determinant of a second rank tensor
    res = 0
    for i, j, k in product(range(3), repeat=3):
        res += levi[i][j][k] * a[i][0] * a[j][1] * a[k][2]
    return res


