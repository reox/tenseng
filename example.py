#!/usr/bin/env python3
from tenseng.tenseng import Vector, dyad, trace, sym, skew

x = Vector(1,2,3)
y = Vector(-7, 8,9)

# Addition
print("addition")
z = x + y
# gives -6, 10, 12
print(z[0], z[1], z[2])

# subtraction
print("subtraction")
z = x - y
# gives 8, -6, -6
print(z[0], z[1], z[2])

# Cross product
print("cross product")
z = x @ y
# gives -6, -30, 22
print(z[0], z[1], z[2])

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
print(z[0][0], z[0][1], z[0][2])
print(z[1][0], z[1][1], z[1][2])
print(z[2][0], z[2][1], z[2][2])


e1 = Vector(1, 0, 0)
e2 = Vector(0, 1, 0)
e3 = e1 @ e2

# Volume product
V = (e1 @ e2) * e3
print(V)

print(e3[0], e3[1], e3[2])

print("identity tensor by using dyadic product and orthonormal basis")
z = dyad(e1, e1) + dyad(e2, e2) + dyad(e3, e3)
print(z[0][0], z[0][1], z[0][2])
print(z[1][0], z[1][1], z[1][2])
print(z[2][0], z[2][1], z[2][2])


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
print(s[0][0], s[0][1], s[0][2])
print(s[1][0], s[1][1], s[1][2])
print(s[2][0], s[2][1], s[2][2])

print()
print(w[0][0], w[0][1], w[0][2])
print(w[1][0], w[1][1], w[1][2])
print(w[2][0], w[2][1], w[2][2])
print()

print(v[0][0], v[0][1], v[0][2])
print(v[1][0], v[1][1], v[1][2])
print(v[2][0], v[2][1], v[2][2])


