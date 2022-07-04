import pytest
import numpy as np
import numpy.testing as npt
try:
    from singular_1D import (
        UniformMesh, ShishkinMesh, Solution
        )
except ImportError:
    pass


def test_import():
    from singular_1D.singular_1D import Solution

def test_u_ex():
    mesh = ShishkinMesh(10, 0.1)
    sol = Solution(mesh)
    assert sol.u_ex(0) == 1 and sol.u_ex(1) == 1

def test_working_solve():
    mesh = UniformMesh(10, 0.1)
    sol = Solution(mesh)
    sol.solve()
    npt.assert_almost_equal(sol.U,
                            np.array([1., 0.59951124, 0.44926686,
                                      0.42414467, 0.46158358, 0.53030303,
                                      0.61466276, 0.70684262, 0.80293255,
                                      0.90097752, 1.]),
                            decimal=6)

def test_edge_case_solve():
    mesh = UniformMesh(2, 0.1)
    sol = Solution(mesh)
    sol.solve()
    npt.assert_almost_equal(sol.U,
                            np.array([1., 0.5, 1.]),
                            decimal=6)

def test_compute_error_sol_not_yet_computed():
    mesh = UniformMesh(10, 0.1)
    sol = Solution(mesh)
    with pytest.raises(ValueError):
        sol.compute_error()

def test_working_compute_error():
    mesh = UniformMesh(10, 0.1)
    sol = Solution(mesh)
    sol.solve()
    sol.compute_error()
    npt.assert_almost_equal(sol.error,
                            0.1316604998072,
                            decimal=6)
