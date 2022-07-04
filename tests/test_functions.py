import pytest
import numpy as np
import numpy.testing as npt
try:
    from singular_1D import (
        UniformMesh, ShishkinMesh, Solution, convergence_rates, solve_grid
        )
except ImportError:
    pass


def test_empty_solve_grid():
    Ns = []
    epsilons = []
    assert not len(solve_grid(Ns, epsilons, "uniform"))


def test_wrong_mesh_solve_grid():
    Ns = [16, 32]
    epsilons = [0.1, 0.01]
    with pytest.raises(ValueError):
        solve_grid(Ns, epsilons, "")


def test_working_solve_grid():
    Ns = [16, 32]
    epsilons = [0.1, 0.01]
    errors = solve_grid(Ns, epsilons, "uniform")
    npt.assert_almost_equal(errors,
                            np.array([[0.0919629, 0.13600058],
                                     [0.0506141, 0.19848731]]),
                            decimal=6)

def test_empty_convergence_rates():
    Ns = []
    errors = []
    eps_unif_err, rates, expected_rates = convergence_rates(errors, Ns)
    assert not len(eps_unif_err) and not len(rates) and not len(expected_rates)

def test_working_convergence_rates():
    Ns = [16, 32]
    epsilons = [0.1, 0.01]
    errors = solve_grid(Ns, epsilons, "shishkin")
    eps_unif_err, rates, expected_rates = convergence_rates(errors, Ns)
    flag1 = np.allclose(eps_unif_err, np.array([[0.05599924], [0.03662138]]))
    flag2 = np.allclose(rates[:-1], np.array([[0.61272104]]))
    flag3 = np.allclose(expected_rates, np.array([[0.67807191], [0.73696559]]))

    assert flag1 and flag2 and flag3
