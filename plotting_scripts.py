"""All the scripts used to generate the plots from the report."""

import numpy as np
import matplotlib.pyplot as plt
from singular_1D.singular_1D import (
    UniformMesh, ShishkinMesh, Solution,
    solve_grid, convergence_rates, plot_errors
    )


"""Plot approx U vs exact u on uniform mesh."""
u = UniformMesh(16, 0.1)
sol = Solution(u)
sol.solve()
plt.plot(sol.mesh, sol.U, "-o", alpha=0.35, label=f"{sol.mesh}")

u = UniformMesh(64, 0.1)
sol = Solution(u)
sol.solve()
plt.plot(sol.mesh, sol.U, "-o", alpha=0.35, label=f"{sol.mesh}")

plt.title(f"Approximate vs exact solution for ϵ = {sol.eps}")
plt.plot(np.linspace(0, 1, 1024), sol.u_ex(np.linspace(0, 1, 1024)),
         alpha=1, label="exact solution")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.show()

u = UniformMesh(16, 0.001)
sol = Solution(u)
sol.solve()
plt.plot(sol.mesh, sol.U, "-o", alpha=0.35, label=f"{sol.mesh}")

u = UniformMesh(64, 0.001)
sol = Solution(u)
sol.solve()
plt.plot(sol.mesh, sol.U, "-o", alpha=0.35, label=f"{sol.mesh}")

plt.title(f"Approximate vs exact solution for ϵ = {sol.eps}")
plt.plot(np.linspace(0, 1, 1024), sol.u_ex(np.linspace(0, 1, 1024)),
         alpha=1, label="exact solution")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.show()


"""Compare solution on S-mesh with solution on U-mesh for small N"""
m = ShishkinMesh(32, 0.01)
sol = Solution(m)
sol.solve()
sol.plot_sol()

m = UniformMesh(32, 0.01)
sol = Solution(m)
sol.solve()
sol.plot_sol()


"""Compute and compare error for various N and eps"""
Ns = [2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13]
epsilons = [0.01, 0.001, 0.0001]

errors = solve_grid(Ns, epsilons, "shishkin")
plot_errors(errors, Ns, epsilons, "shishkin")

errors = solve_grid(Ns, epsilons, "uniform")
plot_errors(errors, Ns, epsilons, "uniform")

"""Visually investigate why the error for the uniform mesh increases with N."""
eps = 0.001
Ns = [2**5, 2**7, 2**9, 2**11]
for N in Ns:
    u = UniformMesh(N, eps)
    sol = Solution(u)
    sol.solve()
    sol.plot_sol()


"""Compute and plot convergence rate for Shishkin mesh."""
Ns = [2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13]
epsilons = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

errors = solve_grid(Ns, epsilons, "shishkin")
_, rates, expected_rates = convergence_rates(errors, Ns)
print(f"\nTheoretical convergence rate for each N \n{expected_rates}")
print(f"\nConvergence rate for each N \n{rates}")

plt.semilogx(Ns[:-1], rates[:-1], label="r^N")
plt.semilogx(Ns[:-1], expected_rates[:-1], label="expected rate")
plt.title("Convergence rate for various N")
plt.xlabel("N")
plt.ylabel("error")
plt.legend()
plt.show()
