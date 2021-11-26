#!/usr/bin/env python3
from tenseng.tenseng import Vector, dyad, trace, sym, skew

x = Vector(1,2,3)
y = Vector(-7, 8,9)

# Addition
print("addition")
z = x + y
# gives -6, 10, 12
print(z)

# subtraction
print("subtraction")
z = x - y
# gives 8, -6, -6
print(z)

# Cross product
print("cross product")
z = x @ y
# gives -6, -30, 22
print(z)

# scalar product
print("scalar product")
z = x * y
# gives 36
print(z)

# dyad
print("dyadic product")
z = dyad(x, y)
# gives:
# -7 8 9
# -14 16 18
# -21 24 27
print(z)


e1 = Vector(1, 0, 0)
e2 = Vector(0, 1, 0)
e3 = e1 @ e2

print("Volume Product of (e1 x e2) . e3")
# Volume product
V = (e1 @ e2) * e3
print(V)

print(e3)

print("identity tensor by using dyadic product and orthonormal basis")
z = dyad(e1, e1) + dyad(e2, e2) + dyad(e3, e3)
print(z)


print("trace")
print(trace(z))

# Trace: tr(dyad(a,b)) = a dot b
print("identity")
z = trace(dyad(x, y))
print(z)
z = x * y
print(z)

print("sym and skew")
z = dyad(x, y)

s = sym(z)
w = skew(z)
v = s + w
print(v, " = ", s, " + ", w)

# The skew part is traceless:
print("trace of skew(z) == 0")
print(trace(w))


