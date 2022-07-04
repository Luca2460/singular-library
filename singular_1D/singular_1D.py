"""Library to numerically solve ϵ*u'' + u' = 1, u(0)=u(N)=1."""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


class BaseMesh:
    """Abstract base class for a mesh with N + 1 points.

    ...

    Attributes
    ----------
    mesh : list
        The list of the mesh points.
    """

    def __init__(self):
        pass

    def h(self, i: int) -> float:
        """Return h_i := x_i - x_{i-1}.

        Return the step h_i using its definition h_i := x_i - x_{i-1}.

        Parameters
        ----------
        i : int
            The index of the step h_i requested.

        Returns
        -------
        float
            The step h_i = x_i - x_{i-1}
        """

        if i > 0 and i < len(self.mesh):
            return self[i] - self[i-1]

        else:
            raise ValueError("i must be an index between 1 and "
                             f"{len(self.mesh)}")

    def __str__(self):
        return f"{self.name} mesh, N={self.N}"

    def __iter__(self):
        return iter(self.mesh)

    def __len__(self):
        return len(self.mesh)

    def __getitem__(self, index):
        return self.mesh[index]


class UniformMesh(BaseMesh):
    """Create a uniform mesh of N + 1 points."""

    name = "uniform"

    def __init__(self, N: int, eps: float) -> None:
        """Create uniform mesh x_i = i/N , i = 0,...,N.

        Create uniform mesh with N+1 equally spaced points in [0,1]

        Parameters
        ----------
        N : int
            The N of the mesh. N must be larger than 1
            so that the mesh has internal points. The mesh has N+1 points.

        eps : float
            The value of ϵ for the problem to be solved on this mesh.
        """

        if N > 1:
            self.N = N
        else:
            raise ValueError("N must be larger than 1.")

        if eps > 0:
            self.eps = eps
        else:
            raise ValueError("eps must be positive.")

        self.mesh = np.linspace(0, 1, N + 1)


class ShishkinMesh(BaseMesh):
    """Create a Shishkin piecewise-uniform mesh of N + 1 points."""

    name = "shishkin"

    def __init__(self, N: int, eps: float, C: float = 1) -> None:
        """Create Shishkin mesh with N+1 points.

        Create Shishkin mesh:
        {x_i = ih | i = 0, ..., N/2 , x_i = σ + iH | i = N/2 + 1, ..., N}
        With:
        h = 2*σ / N
        H = 2*(1 - σ) / N
        σ = min(eps*C*ln N, 1/2)

        Parameters
        ----------
        N : int
            The N of the mesh. N must be larger than 1
            so that the mesh has internal points. The mesh has N+1 points.

        eps : float
            The value of ϵ for the problem to be solved on this mesh.

        C : float
            The constant C_σ for the transition point σ.
            It is raccomended that C >= 1.
        """

        if N > 1:
            self.N = N
        else:
            raise ValueError("N must be larger than 1.")

        if eps > 0:
            self.eps = eps
        else:
            raise ValueError("eps must be positive.")

        self.sigma = min(eps*C*np.log(N), 1/2)

        fine_mesh = np.linspace(0, self.sigma, N//2, endpoint=False)
        coarse_mesh = np.linspace(self.sigma, 1, N//2 + 1)
        self.mesh = np.concatenate((fine_mesh, coarse_mesh))


class Solution:
    """Create Solution object.

    ...

    Attributes
    ----------
    mesh : list
        The list of the mesh point over which the discretized problem
        is meant to be solved.
    N : int
        The value of N of the mesh. The mesh contains N + 1 points.
    eps : float
        The value of ϵ for the problem to be solved.
    U : list
        The approximate solution computed at the mesh points.
    error : float
        The maximum error over all the mesh points.

    """

    def __init__(self, mesh) -> None:
        """Initialize solution class on input mesh."""

        self.mesh = mesh
        self.N = mesh.N
        self.eps = mesh.eps
        self.U = None
        self.error = None

    def u_ex(self, x: float) -> float:
        """Exact solution u of eps * u'' + u' = 1, u(0)=u(N)=1.

        Analytical solution u(x) for the problem ϵ*u'' + u' = 1, u(0)=u(N)=1.

        Parameters
        ----------
        x : float
            The value at which to evaluate the exact solution of the problem.

        Returns
        -------
        float
            The exact solution evaluated at x.
        """

        def w(x): return np.exp(-x/self.eps)

        return x + (w(x) - w(1)) / (1 - w(1))

    def solve(self) -> None:
        """Compute approximate solution U."""
        # Compute coefficient matrix of M*U = b; U approx. solution.
        M = self._coefficient_matrix()

        # Initialize RHS of the system.
        b = [1] * (self.N - 1)

        # Apply BC U(0)=U(1)=1 by modifying the RHS of the system.
        b[0] = 1 - self.eps / (self.mesh.h(1)*self.mesh.h(2))
        b[-1] = 1 - self.eps / self.mesh.h(self.N)**2 - 1 / self.mesh.h(self.N)

        # Solve the system and append values U_0 and U_N known from BC.
        U = np.linalg.solve(M, b)
        self.U = np.concatenate((np.array((1,)), U, np.array((1,))))

    def plot_sol(self) -> None:
        """Plot approximate solution and exact one."""
        try:
            plt.plot(self.mesh, self.u_ex(self.mesh.mesh),
                     "-*", alpha=0.5, label="exact")
            plt.plot(self.mesh, self.U, "-o", alpha=0.5, label="approximate")
            plt.title(f"Approx. solution for ϵ = {self.eps} on {self.mesh}")
            plt.xlabel("x")
            plt.ylabel("u")
            plt.legend()
            plt.show()

        except ValueError:
            print("Solution not yet computed.")
            return

    def compute_error(self) -> None:
        """Compute error for approximate solution."""
        try:
            self.error = max(abs(self.u_ex(self.mesh.mesh) - self.U))

        except TypeError:
            raise ValueError("Solution not yet computed,"
                             "could not compute error.")

    def _coefficient_matrix(self) -> np.ndarray:
        """Generate coefficient matrix for discretization of ϵ*u'' + u' = 1.

        Returns
        -------
        numpy.ndarray
            The coefficient matrix for the discretization of the problem.
        """
        main_diag = [- self.eps / (self.mesh.h(i+1) ** 2)
                     - self.eps / (self.mesh.h(i+1) * self.mesh.h(i))
                     - 1 / self.mesh.h(i+1)
                     for i in range(1, self.N)]

        sub_diag = [self.eps / (self.mesh.h(i+1) * self.mesh.h(i))
                    for i in range(2, self.N)]

        super_diag = [self.eps / self.mesh.h(i+1) ** 2
                      + 1 / self.mesh.h(i+1)
                      for i in range(1, self.N - 1)]

        return (np.diag(main_diag)
                + np.diag(sub_diag, -1)
                + np.diag(super_diag, 1))


def solve_grid(Ns: list, epsilons: list, mesh: str) -> np.ndarray:
    """Compute errors for all the combinations of N and eps provided.

    Computes error for each eps in epsilons and each N in Ns.

    Parameters
    ----------
    Ns : list[int]
        The list containing the desired values of N for which
        to compute the errors.
    epsilons : list[float]
        The list containing the desired values of eps for which
        to compute the errors.

    Returns
    -------
    numpy.ndarray
        The values of the errors obtained for each combination of N and eps.
    """
    errors = np.zeros((len(Ns), len(epsilons)))

    for i, n in enumerate(Ns):
        for j, epsilon in enumerate(epsilons):
            if mesh == "uniform":
                m = UniformMesh(n, epsilon)

            elif mesh == "shishkin":
                m = ShishkinMesh(n, epsilon)

            else:
                raise ValueError(f"inserted mesh: \"{mesh}\" not available")

            sol = Solution(m)
            sol.solve()
            sol.compute_error()
            errors[i][j] = sol.error

    return errors


def convergence_rates(errors: np.ndarray, Ns: list) -> Tuple[np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray]:
    """Compute convergence rate of eps-uniform errors for each N.

    Compute maximum error across all eps for each N,
    as well as numerical and theoretical error's convergence rate.

    Parameters
    ----------
    errors: numpy.ndarray
        The matrix containing the error for various Ns and eps.
    Ns : list[int]
        The list of N used to compute the errors. Each value of Ns
        is expected to be double of its preceding one.

    Returns
    -------
    numpy.ndarray
        The maximum error across all eps for each N.
    numpy.ndarray
        The experimental convergence rate.
    numpy.ndarray
        The theoretical convergence rate.
    """
    if len(Ns) > 0:
        # Find max error across all eps for a given value of N.
        eps_uniform_err = np.array([max(error_N) for error_N in errors])

        rates = np.array(
                         [np.log2(eps_uniform_err[i]/eps_uniform_err[i+1])
                          for i in range(len(errors) - 1)]
                         + [0]
                        )

        expected_rates = np.array([1 - np.log2(1+np.log(2)/np.log(N))
                                   for N
                                   in Ns])

        return (eps_uniform_err.reshape(-1, 1),
                rates.reshape(-1, 1),
                expected_rates.reshape((-1, 1)))

    else:
        return [], [], []


def plot_errors(errors: np.ndarray,
                Ns: list,
                epsilons: list,
                mesh: str) -> None:
    """Plot error vs N for various N and eps.

    Parameters
    ----------
    errors : numpy.ndarray
        The matrix containing the error for various Ns and eps.
    Ns : list[int]
        The list of N used to compute the errors.
    epsilons : list[float]
        The list of eps used to compute the errors.
    mesh : str
        The name of the mesh used to compute the approximate solution.
    """
    f, ax = plt.subplots(1)

    for i, eps in enumerate(epsilons):
        plt.loglog(Ns, errors[:, i], "-o", label=f"eps = {eps}")

    plt.loglog(Ns, 0.5*np.log(Ns)/Ns, "-*", label="const * N^-1 log N")
    plt.loglog(Ns, 5/np.array(Ns), "-x", label="N^-1")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    plt.title(f"Error vs N for {mesh} mesh")
    plt.xlabel("N")
    plt.ylabel("error")
    plt.legend()
    plt.show()
