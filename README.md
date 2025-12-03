# maxwellgp

**maxwellgp** — GP framework for modeling 3D time harmonic electromagnetic fields constrained by Maxwell’s equations 

## Description

`maxwellgp` is a Python package for building Gaussian Process (GP) models of electromagnetic fields (E, B) that **automatically satisfy the time-harmonic Maxwell equations**.  



## Features

- ✨ **Physically consistent GPs**: ensures fields obey Maxwell’s equations by design (not just approximate).  

## Requirements / Dependencies

- Python 3.x  
- JAX (for differentiable GP and kernels)  
- Numpy / SciPy  


## Installation

```bash
git clone https://github.com/Felixiose/maxwellgp.git
cd maxwellgp
pip install -e .
