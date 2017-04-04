class Vector(object):
    """
    Vector Object for tensors of at least 1. rank
    Scalar values (tensors 0. rank) can be used as scalars

    TODO: we actually need to have 0. rank tensors...
    TODO: we need to define the rank somehow?!
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

        self.data[key] = value

    def __add__(self, other):
        res = Vector(dim=self.dim)
        for i in range(self.dim):
            res[i] = self[i] + other[i]

        return res

    def __sub__(self, other):
        res = Vector(dim=self.dim)
        for i in range(self.dim):
            res[i] = self[i] - other[i]

        return res

    # TODO how we can multiply by scalar?
    def __mul__(self, other):
        # skalarproduct
        res = 0
        for i in range(self.dim):
            res += self[i] * other[i]

        return res

    def __matmul__(self, other):
        # cross product
        res = Vector(dim=self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    res[i] += levi[i][j][k] * self[j] * other[k]

        return res


# TODO need to define for higher orders as well
def dyad(a, b):
    res = Vector(Vector(), Vector(), Vector())
    for i in range(3):
        for j in range(3):
            res[i][j] = a[i] * b[j]
    return res


def trace(a):
    res = 0
    for i in range(3):
        res += a[i][i]
    return res


# TODO need to define for higher orders... A_ijkl => A_klij
def transpose(a):
    res = Vector(Vector(), Vector(), Vector())
    for i in range(3):
        for j in range(3):
            res[i][j] = a[j][i]
    return res


def sym(a):
    # 1/2 * (a + aT)
    a = a + transpose(a)
    res = Vector(Vector(), Vector(), Vector())
    for i in range(3):
        for j in range(3):
            res[i][j] = 0.5 * a[i][j]
    return res


def skew(a):
    # 1/2 * (a - aT)
    a = a - transpose(a)
    res = Vector(Vector(), Vector(), Vector())
    for i in range(3):
        for j in range(3):
            res[i][j] = 0.5 * a[i][j]
    return res


def vol(a):
    pass


def dev(a):
    pass


# Kroneker Delta
kron = Vector(Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1))

# Levi Civita
levi = Vector(Vector(Vector(0, 0, 0), Vector(0, 0, 1), Vector(0, -1, 0)),
              Vector(Vector(0, 0, -1), Vector(0, 0, 0), Vector(1, 0, 0)),
              Vector(Vector(0, 1, 0), Vector(-1, 0, 0), Vector(0, 0, 0)))
