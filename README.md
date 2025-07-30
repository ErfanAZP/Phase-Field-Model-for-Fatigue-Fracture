# Phase Field Fatigue Fracture Model 

DESCRIPTION:
This code implements a phase field method for simulating high-cycle fatigue 
fracture in materials. The implementation is based on an adaptive switching algorithm 
that alternates between Newton-Raphson and Modified Newton-Raphson solution schemes 
to optimize computational efficiency while maintaining accuracy.

This code benchmarks the fatigue crack propagation in a 2D single edge notched tension (SENT) 
specimen under displacement-controlled loading conditions. The phase field approach 
allows for natural crack initiation and propagation without predefined crack paths. 

KEY FEATURES:
- Phase field modeling of fatigue fracture mechanics
- Adaptive switching between Newton-Raphson and Modified Newton-Raphson solvers allowing faster simulation for HCF problems
- Multiple fatigue degradation functions based on various literature 
- Various strain decomposition strategies (Isotropic, VolDev, Spectral, NoTension)
- Support for monotonic, cyclic, and constant load amplitude fatigue loading (both displacement and force controlled)
- 2D plane stress and plane strain analysis capabilities
- VTK output for visualization in ParaView
- CSV output for post-processing and analysis

CITATION:
If you use this code in your research, please cite the following paper:

Azinpour, E., Gil, J., Darabi, R., De Jesus, A., Reis, A., & De Sá, J. C. (2025). 
High-cycle fatigue analysis of laser-based directed energy deposition maraging steels: 
Combined phase field and experimental studies. International Journal of Fatigue, 198, 108970. 
https://doi.org/10.1016/j.ijfatigue.2025.108970

DEPENDENCIES:
- Ferrite.jl (Finite element framework)
- FerriteMeshParser.jl (Mesh input/output)
- SparseArrays.jl (Sparse matrix operations)
- Tensors.jl (Tensor operations)
- SuiteSparse.jl (Linear solvers)
- CSV.jl, DataFrames.jl (Data output)
- Plots.jl (Plotting)
- WriteVTK.jl (VTK output for visualization)

USAGE:
1. Ensure all dependencies are installed
2. Prepare mesh file ("sent.inp") with appropriate boundary sets*
3. Configure simulation parameters in the CONFIG struct
4. Run: julia pfm_fatigue.jl

* Please note that Abaqus .inp file is used in this file, while alternative mesh formats such as GMSH (.msh) will also be supported via Ferrite.jl
More information regarding FEM implementation using Ferrite.jl can be found in their official webpage:
https://ferrite-fem.github.io/Ferrite.jl/stable/

