import pytest
import numpy as np
import numpy.testing as npt
try:
    from singular_1D import (
        UniformMesh, ShishkinMesh
        )
except ImportError:
    pass


def test_import():
    from singular_1D.singular_1D import BaseMesh, UniformMesh, ShishkinMesh


def test_UniformMesh_init_wrong_N():
    with pytest.raises(ValueError):
        mesh = UniformMesh(0, 0.1)

def test_UniformMesh_init_wrong_eps_0():
    with pytest.raises(ValueError):
        mesh = UniformMesh(10, 0)

def test_UniformMesh_init_wrong_eps_neg():
    with pytest.raises(ValueError):
        mesh = UniformMesh(10, -0.1)

def test_working_UniformMesh():
    mesh = UniformMesh(10, 0.01)
    npt.assert_almost_equal(mesh.mesh,
                            np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5,
                                      0.6, 0.7, 0.8, 0.9, 1.]),
                            decimal=6)


def test_ShishkinMesh_init_wrong_N():
    with pytest.raises(ValueError):
        mesh = ShishkinMesh(0, 0.1)

def test_ShishkinMesh_init_wrong_eps_0():
    with pytest.raises(ValueError):
        mesh = ShishkinMesh(10, 0)

def test_ShishkinMesh_init_wrong_eps_neg():
    with pytest.raises(ValueError):
        mesh = ShishkinMesh(10, -0.1)

def test_working_UniformMesh():
    mesh = ShishkinMesh(10, 0.1)
    npt.assert_almost_equal(mesh.mesh,
                            np.array([0., 0.0460517, 0.0921034, 0.13815511,
                                      0.18420681, 0.23025851, 0.38420681,
                                      0.53815511, 0.6921034, 0.8460517, 1.]),
                            decimal=6)


def test_h_input_too_small():
    mesh = UniformMesh(10, 0.1)
    with pytest.raises(ValueError):
        mesh.h(0)

def test_h_input_too_large():
    mesh = UniformMesh(10, 0.1)
    with pytest.raises(ValueError):
        mesh.h(11)


# fai testing mettendo corner cases come test cases,
# esempio matrici molto piccole o con entry vuote
# Oppure fai anche testing dei singoli metodi e delle funzioni
# Puoi ispirarti alle cose di testing che fa il prof per l'ex di gruppi