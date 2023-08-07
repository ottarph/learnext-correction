import dolfin as df
import numpy as np

from data_prep.clement.clement import *

mesh = df.UnitSquareMesh(60, 60)

p = 3
V = df.FunctionSpace(mesh, "CG", p)
V2 = df.VectorFunctionSpace(mesh, "CG", p, 2)
u = df.Function(V)

u = df.interpolate(df.Expression("(3*x[0] + 4*x[1])*1.0", element=V.ufl_element()), V)
u_grad = df.interpolate(df.Expression(("(3)*1.0", "(4)*1.0") , element=V2.ufl_element()), V2)

print(f"{df.norm(u) = }")
print(f"{df.norm(u_grad) = }")

ci, CI = clement_interpolate(df.grad(u), with_CI=True)

print(f"{df.norm(ci) = }")
print(f"{df.errornorm(ci, u_grad) = }")

u_tmp = df.interpolate(df.Expression("(3*x[0] + 4*x[1])*2.0", element=V.ufl_element()), V)
du_tmp = df.interpolate(df.Expression(("(3)*2.0", "(4)*2.0") , element=V2.ufl_element()), V2)
u.vector().set_local(u_tmp.vector().get_local())
u_grad.vector().set_local(du_tmp.vector().get_local())
print(f"{df.errornorm(CI(), u_grad) = }")
u_tmp = df.interpolate(df.Expression("(3*x[0] + 4*x[1])*3.0", element=V.ufl_element()), V)
du_tmp = df.interpolate(df.Expression(("(3)*3.0", "(4)*3.0") , element=V2.ufl_element()), V2)
u.vector().set_local(u_tmp.vector().get_local())
u_grad.vector().set_local(du_tmp.vector().get_local())
print(f"{df.errornorm(CI(), u_grad) = }")


u = df.interpolate(df.Expression("(3*x[0]*x[0] + 4*x[1]*x[1])*1.0", element=V.ufl_element()), V)
u_grad = df.interpolate(df.Expression(("(6*x[0])*1.0", "(8*x[1])*1.0") , element=V2.ufl_element()), V2)

print(f"{df.norm(u) = }")
print(f"{df.norm(u_grad) = }")

ci, CI = clement_interpolate(df.grad(u), with_CI=True)

print(f"{df.norm(ci) = }")
print(f"{df.errornorm(ci, u_grad) = }")

u_tmp = df.interpolate(df.Expression("(3*x[0]*x[0] + 4*x[1]*x[1])*2.0", element=V.ufl_element()), V)
du_tmp = df.interpolate(df.Expression(("(6*x[0])*2.0", "(8*x[1])*2.0") , element=V2.ufl_element()), V2)
u.vector().set_local(u_tmp.vector().get_local())
u_grad.vector().set_local(du_tmp.vector().get_local())
print(f"{df.errornorm(CI(), u_grad) = }")
u_tmp = df.interpolate(df.Expression("(3*x[0]*x[0] + 4*x[1]*x[1])*3.0", element=V.ufl_element()), V)
du_tmp = df.interpolate(df.Expression(("(6*x[0])*3.0", "(8*x[1])*3.0") , element=V2.ufl_element()), V2)
u.vector().set_local(u_tmp.vector().get_local())
u_grad.vector().set_local(du_tmp.vector().get_local())
print(f"{df.errornorm(CI(), u_grad) = }")


for _ in range(100):
    print(f"{df.norm(CI())}")
    tmp = u.vector().get_local()
    tmp *= 1.05
    u.vector().set_local(tmp)
