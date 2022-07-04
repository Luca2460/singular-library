# singular-library
Project for Oxford MMSC course "Python in scientific computing".

The library contains code to solve the 1-D singularly perturbed ODE eps * u'' + u' = 1 , u(0) = u(1) = 1 for the small parameter eps <<1.

The library contains the UniformMesh, ShishkinMesh and Solution classes that can be used together to compute the solution of the problem in few commands.

The said objects can then be used in more complex functions created by the user. Some functions to test the code for various inputs are already implemented.

The file plotting_scripts.py includes all the scripts used in the report.
