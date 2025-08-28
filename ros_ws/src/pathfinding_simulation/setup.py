#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[
        'pathfinding_simulation',
        'pathfinding_simulation.fields',
        'pathfinding_simulation.grid',
        'pathfinding_simulation.utils',
        'pathfinding_simulation.pathfinding'
    ],
    package_dir={'': 'src'}
)

setup(**setup_args)
