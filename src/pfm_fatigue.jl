#=
================================================================================
Phase Field Fatigue Fracture Model 
================================================================================

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

MATERIAL MODELS:
- Brittle fracture with phase field regularization 
- Configurable material parameters (E, ν, Gc, ℓ, etc.)

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
================================================================================
=#
using Ferrite, FerriteMeshParser, SparseArrays, Tensors, Printf, LinearAlgebra
using SuiteSparse, CSV, DataFrames, Plots, WriteVTK

# Enums (define these before the Config struct)
@enum SolutionType NewtonRaphson ModifiedNewtonRaphson
@enum SolverMode Standard Adaptive
@enum Load Monotonic Fatigue CLA_U CLA_F  
@enum StrainDecomp Isotropic VolDev Spectral NoTension
@enum FatigueDegMode f0 f1 f2 f3 f4
@enum StressState PlaneStrain PlaneStress

"""
Configuration for the phase field fatigue simulation.
"""
struct Config
    # Simulation parameters
    n_timesteps::Int
    plot_frequency::Int
    history_output_frequency::Int
    
    # Loading parameters
    F_max::Float64
    F_min::Float64
    u_max::Float64
    u_min::Float64
    area::Float64
    
    # Material properties
    E::Float64
    ν::Float64
    Gc::Float64
    ℓ::Float64
    κ::Float64
    αₜ::Float64
    n::Float64
    thck::Float64
    
    # Algorithm parameters and simulation modes
    loadingtype::Load
    fmode::FatigueDegMode
    dmode::StrainDecomp
    n_c::Int
    nᵢ::Int
    rebuild_frequency_base::Int
    CyclesPerIncrement::Float64
    smode::StressState
    dΨ_tol::Float64
    
    # Geometry and output
    a₀::Float64  # Initial crack length
    CrackDirection::Int # Crack direction along x and y axes
    outputset::String
    solverscheme::SolverMode 
    max_crack_length::Float64  
end

# Create a global configuration object
const CONFIG =  Config(
    80000,      # n_timesteps
    500,        # plot_frequency
    500,        # history_output_frequency
    0.0,        # F_max
    0.0,        # F_min
    0.0005,     # u_max
    0.00005,    # u_min
    1.0,        # area
    210000.0,   # E
    0.3,        # ν
    2.7,        # Gc
    0.016 ,     # ℓ
    0.5,        # κ
    62.5,       # αₜ
    0.5,        # n
    1.0,        # thck
    CLA_U, # Loading type:: Monotonic, Cyclic, Constant load amplitude (displacement and load controlled)
    f0,         # Fatigue degradation function mode
    NoTension,  # Strain decomposition strategy mode
    100,        # n_c
    20,         # nᵢ
    3,          # rebuild_frequency_base
    1.0,        # CyclesPerIncrement
    PlaneStrain,# Stress State of the problem
    3.0,        # dΨ_tol
    0.0,        # a₀
    1,          # CrackDirection
    "TOP",      # outputset
    Adaptive,   # Solver method
    0.49        # Maximum crack length  to halt the simulation
)

# Structs
struct Brittle{T, S <: SymmetricTensor{4, 3, T}}
    E::T  # Young's modulus
    ν::T  # Poisson's ratio
    G::T  # Shear modulus
    K::T  # Bulk modulus
    Gc::T # Fracture Toughness
    ℓ::T  # Phase field length scale
    dmode::StrainDecomp # Strain Decomposition mode
    Dᵉ::S # Elastic stiffness tensor
    load::Load  #Loading type
    dim::Int64  #Number of dimensions
    l_r::T #Load ratio
    fmode::FatigueDegMode  #Fatigue degradation function mode
    n::T  #Material parameter (exponent)
    κ::T   #Material parameter (exponent for logarithmic degradation function)
    αₜ::T  #Material parameter (history parameter threshold)
    smode::StressState  #Stress state mode (plane stress and plane strain)
end

struct MaterialState{T}
    H::T # History variable
    ϕ::T # Phase field variable from last increment
    α::T # AccumulatedFatigue
    ψ::T # Strain energy from last increment 
    Ψ_eff::T  # Effective strain energy
end

mutable struct SolverState{T,F}
    rebuild_counter::T
    rebuild_frequency_base::T
    rebuild_frequency_min::T
    rebuild_start::Bool
    no_restart::T
    strategy::SolutionType
    P_max::F
    P_min::F
    u_max::F
    u_min::F
    area::F
    thck::F
    timesteps::T
    nᵢ::T
    NewtonTOL::F
    dΨ_tol::F      # Tolerance for alpha changes
    loading::Load
    Ψ_eff_new::F   # Store current timestep's alpha
    Ψ_eff_old::F  # Store previous timestep's alpha
    current_n_c::T  #  field to track current n_c
    base_n_c::T     #  original n_c
    adaptive_mode::Bool  #  Activation mode for adaptive switching scheme 
end

mutable struct OutputVariables{T}
    plotframe::T
    totalIterations_outer::T
    totalIterations_phi::T
    totalIterations_u::T
    matrixFactorizations_u::T
    matrixFactorizations_phi::T
    plotFrequency::T
    historyFrequency::T
    a0::Float64
    CrackDirection::T
    OutputSet::String
end

# Constructor functions
function Brittle(E, ν, Gc, ℓ, dmode, load, dim, l_r, fmode, n, κ, αₜ, smode)
    δ(i,j) = i == j ? 1.0 : 0.0 # helper function
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)

    temp = if smode == PlaneStress
        (i,j,k,l) -> 2.0G * (0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) + ν/(1.0-ν)*δ(i,j)*δ(k,l))
    else  # plane_strain
        (i,j,k,l) -> 2.0G * (0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) + ν/(1.0-2.0ν)*δ(i,j)*δ(k,l))
    end
    Dᵉ = SymmetricTensor{4, 3}(temp)
    return Brittle(E, ν, G, K, Gc, ℓ, dmode, Dᵉ, load, dim, l_r, fmode, n, κ, αₜ, smode)
end

function MaterialState()
    return MaterialState(0.0, 0.0, 0.0, 0.0, 0.0)
end

function SolverState(RebuildFrequencyBase, RebuildPrN, P_max, P_min, u_max, u_min, area, thck, nT, nᵢ, NewtonTOL, dΨ_tol, load)
    return SolverState(0, RebuildFrequencyBase, RebuildPrN, true, 0, ModifiedNewtonRaphson, P_max, P_min, u_max, u_min, area, thck, nT, nᵢ, NewtonTOL, dΨ_tol,  load, 0.0, 0.0,
                         RebuildPrN, RebuildPrN, false)
end

function OutputVariables(field_frequency, history_frequency, a0, CrackDirection, outputset)
    return OutputVariables(0, 0, 0, 0, 0, 0, field_frequency, history_frequency, a0, CrackDirection, outputset)
end

# Helper functions
function symmetrize_lower!(K)
    for i in 1:size(K,1)
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end

function CrackDrive(σ::SymmetricTensor{2,3,Float64}, ε::SymmetricTensor{2,3,Float64}, mat::Brittle)
    dmode = mat.dmode
    smode = mat.smode

    E, ν = mat.E, mat.ν
    if smode == PlaneStress
        λ = E * ν / (1 - ν^2)
        μ = E / (2 * (1 + ν))
        K = E / (2 * (1 - ν))  # Plane stress bulk modulus
    else  # PlaneStrain
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        K = E / (3 * (1 - 2ν))
    end
    
    if dmode == Isotropic
        Psi = 0.5 * σ ⊡ ε
    elseif dmode == VolDev
        Psi = μ * dev(ε) ⊡ dev(ε)
        if tr(ε) > 0
            Psi += 0.5 * K * tr(ε)^2
        end
    elseif dmode == Spectral
        εₚ = eigvals(ε)
        Psi = sum(εₚ) > 0 ? λ/2 * sum(εₚ)^2 : 0.0
        for e in εₚ
            Psi += e > 0 ? μ * e^2 : 0.0
        end
    elseif dmode == NoTension
        εₚ = sort(eigvals(ε))
        
        if εₚ[1] > 0
            Psi = λ/2 * sum(εₚ)^2 + μ * sum(εₚ.^2)
        elseif ν * εₚ[1] + εₚ[2] > 0
            Psi = λ/2 * (εₚ[3] + εₚ[2] + 2ν * εₚ[1])^2 + μ * ((εₚ[3] + ν * εₚ[1])^2 + (εₚ[2] + ν * εₚ[1])^2)
        elseif (1 - ν) * εₚ[3] + ν * (εₚ[1] + εₚ[2]) > 0
            Psi = λ/2 * (1 + ν) / (ν * (1 - ν^2)) * ((1 - ν) * εₚ[3] + ν * εₚ[1] + ν * εₚ[2])^2
        else
            Psi = 0.0
        end
    else
        error("Unsupported decomposition mode")
    end
    return Psi 
end

function CrackTrack(q::Vector, dh::DofHandler, CellValues::CellValues, grid::Grid, a0::Float64, CrackDirection::Int64)
    Ac = a0
    v = CrackDirection
    for (i, cell) in enumerate(CellIterator(dh))
        reinit!(CellValues, cell)
        eldofs = celldofs(cell)
        ϕe = q[eldofs]
        if maximum(ϕe) >= 0.95
            node_coords = getcoordinates(grid, i)
            for q_point in 1:getnquadpoints(CellValues)
                ϕ = function_value(CellValues, q_point, ϕe)
                if ϕ >= 0.95
                    coords = spatial_coordinate(CellValues, q_point, node_coords)
                    Ac = coords[v] > Ac ? coords[v] : Ac
                end
            end
        end
    end
    return Ac
end

function Calculate_SIF(ΔF, Ac)  # Not being used in this benchmark
    W, B = 25.0, 3.0
    α = Ac / W
    f_α = (2 + α) * (0.886 + 4.64*α - 13.32*α^2 + 14.72*α^3 - 5.6*α^4) / (1 - α)^(3/2)
    ΔK_I = (ΔF / (B * sqrt(W))) * f_α
    ΔK_I = ΔK_I * 0.031623 # conversion to MPa√m
    return ΔK_I
end

function fatigue_degradation(α::Float64, material::Brittle)
    κ, αₜ = material.κ, material.αₜ
    mode = material.fmode

    if mode == f0
        return α >= αₜ ? (2αₜ/(αₜ + α))^2 : 1.0
    elseif mode == f1
        if α ≤ αₜ
            return 1.0
        elseif αₜ < α <= αₜ * 10^(1/κ)
            return (1 - κ * log(α / αₜ))^2
        else
            return 0.0
        end
    elseif mode == f2
        return α > 0.0 ? ((1 - α / (α + αₜ))^2) : 1.0
    elseif mode == f3
        return (0 < α < αₜ) ? ((1 - (α / αₜ))^2) : 1.0
    elseif mode == f4 
        if α ≤ αₜ
            return 1.0
        elseif αₜ < α <= αₜ * 10^(1/κ)
            return (κ * log(αₜ / α))^2
        end
    else
        error("Invalid fatigue degradation mode")
    end
end

# Assembly functions
function assemble_element_u!(Ke::Matrix, fe::Vector, cell, cellvalues_u::CellValues, facevalues::FacetValues,
        ue::Vector, material::Brittle, state, state_old, timestep, trac::Vec{2, Float64})
        nbase_u = getnbasefunctions(cellvalues_u)
        #Loop over integration points
        D = material.Dᵉ

        for q_point in 1:getnquadpoints(cellvalues_u)
            #dvol,
            dΩᵤ=getdetJdV(cellvalues_u,q_point)
            #Total strain
            if material.dim == 2
                ε_2D = function_symmetric_gradient(cellvalues_u,q_point,ue)
                if material.smode == PlaneStress
                  ∇ˢu = SymmetricTensor{2,3,Float64}((ε_2D[1,1],ε_2D[1,2], 0.0,
                  ε_2D[2,2], 0.0, ε_2D[1,1]*(-material.ν)))
                else  # plane_strain
                  ∇ˢu = SymmetricTensor{2,3,Float64}((ε_2D[1,1], ε_2D[1,2], 0.0,
                                                      ε_2D[2,2], 0.0, 0.0))
                end
            elseif material.dim == 3
                ∇ˢu =function_symmetric_gradient(cellvalues_u,q_point,ue)
            else
                error("Invalid element dimension")
            end
            #Phase field value
            ϕ = state[q_point].ϕ
            #(undegraded) Stress
            σ=D ⊡ ∇ˢu
            #Strain energy
            Ψ = CrackDrive(σ, ∇ˢu, material)
            #Recover state vairables
            ϕₙ = state_old[q_point].ϕ
            Hₙ = state_old[q_point].H
            αₙ = state_old[q_point].α
            ψₙ = state_old[q_point].ψ
            #Ensure phase field irreversibillity
            if material.load == Monotonic
                Δα = 0.
            elseif material.load ==Fatigue
                Δα = Ψ > ψₙ ? (Ψ-ψₙ) : 0
            elseif material.load == CLA_U || material.load == CLA_F
                eigs = eigvals(∇ˢu)
                ε_max = maximum(eigs)
                ε_min = minimum(eigs)
                ε_m = (ε_max + ε_min) / 2
                σc = (3/16) * sqrt(material.E * material.Gc / (3 * material.ℓ))
                εc = sqrt(material.Gc / (3 * material.ℓ * material.E))
                Ψ_c = 0.5 * σc * εc
                # Calculation of effective accumulative energy
                Ψ_eff = 2 * material.E * (ε_max*(1+material.l_r)/2.0)^2 * ((1 - material.l_r) / 2.0)^material.n
                Δα = Ψ_eff
            end
            α = αₙ + Δα
            #Update state variables
            H = max(Ψ,Hₙ)
            state[q_point]=MaterialState(H,ϕ,α,Ψ,Ψ_eff)
            #Phase field degradation based on old phi
            gdn= (1.0-ϕ)^2 +1.e-7
    
            ##ASSEMBLE DISPLACEMENT PROBLEM
            #Loop over displacement test functions
            for i in 1:nbase_u
    
                if material.dim == 2
                    δε_2d = shape_symmetric_gradient(cellvalues_u,q_point,i)
                    δε = SymmetricTensor{2,3,Float64}((δε_2d[1,1],δε_2d[1,2], 0.0,
                        δε_2d[2,2], 0.0, 0.0))
                else
                    δε =shape_symmetric_gradient(cellvalues_u,q_point,i)
                end
                #Add contribution to element rhs
    
                fe[i] += gdn*(δε ⊡ σ)*dΩᵤ
                #Loop over displacement  trial shape functions
                for j in 1:i
                    if material.dim ==2
                        ε_2d = shape_symmetric_gradient(cellvalues_u,q_point,j)
                        ε = SymmetricTensor{2,3,Float64}((ε_2d[1,1],ε_2d[1,2], 0.0,
                            ε_2d[2,2], 0.0, 0.0))
                    else
                        ε = shape_symmetric_gradient(cellvalues_u,q_point,j)
                    end
                    Ke[i,j] += gdn*δε ⊡ D ⊡ ε * dΩᵤ
                end
            end
        end
        symmetrize_lower!(Ke)
        if material.load == CLA_F
        ∂Ωₜ = cell.grid.facetsets["TOP"]
          ## Apply traction loading                             
          for facet in 1:nfacets(cell)
              if (cellid(cell), facet) ∈ ∂Ωₜ
                reinit!(facevalues, cell, facet)
                  for q_point in 1:getnquadpoints(facevalues)
                    dΓ = getdetJdV(facevalues, q_point)
                    n = getnormal(facevalues, q_point)
                    trac_n = (trac ⋅ n) * n 
                    for i in 1:nbase_u
                        δu = shape_value(facevalues, q_point, i)
                        fe[i] -= δu ⋅ trac_n * dΓ
                    end
                  end
              end
          end
        end
        return Ke, fe
end

function assemble_residual_u!(fe::Vector, cell, cellvalues_u::CellValues, facevalues::FacetValues,
        ue::Vector, material::Brittle, state, state_old, trac::Vec{2, Float64}, store::Bool)
        nbase_u = getnbasefunctions(cellvalues_u)
        #Loop over integration points
        D = material.Dᵉ
        for q_point in 1:getnquadpoints(cellvalues_u)
            #dvol,
            dΩᵤ=getdetJdV(cellvalues_u,q_point)
            #Total strain
            if material.dim == 2
                ε_2D = function_symmetric_gradient(cellvalues_u,q_point,ue)
                if material.smode == PlaneStress
                  ∇ˢu = SymmetricTensor{2,3,Float64}((ε_2D[1,1],ε_2D[1,2], 0.0,
                  ε_2D[2,2], 0.0, ε_2D[1,1]*(-material.ν)))
                else  # plane_strain
                  ∇ˢu = SymmetricTensor{2,3,Float64}((ε_2D[1,1], ε_2D[1,2], 0.0,
                                                      ε_2D[2,2], 0.0, 0.0))
                end
            elseif material.dim == 3
                ∇ˢu =function_symmetric_gradient(cellvalues_u,q_point,ue)
            else
                error("Invalid element dimension")
            end
            #Phase field value
            ϕ = state[q_point].ϕ
            #(undegraded) Stress
            σ=D ⊡ ∇ˢu
            #Strain energy
            Ψ = CrackDrive(σ, ∇ˢu, material)
              #Recover state vairables
              ϕₙ = state_old[q_point].ϕ
              Hₙ = state_old[q_point].H
              αₙ = state_old[q_point].α
              ψₙ = state_old[q_point].ψ
              #Ensure phase field irreversibillity
              if material.load == Monotonic
                  Δα = 0.
              elseif material.load ==Fatigue
                  Δα = Ψ > ψₙ ? (Ψ-ψₙ) : 0
              elseif material.load == CLA_U || material.load == CLA_F
                eigs = eigvals(∇ˢu)
                ε_max = maximum(eigs)
                ε_min = minimum(eigs)
                ε_m = (ε_max + ε_min) / 2
                σc = (3/16) * sqrt(material.E * material.Gc / (3 * material.ℓ))
                εc = sqrt(material.Gc / (3 * material.ℓ * material.E))
                Ψ_c = 0.5 * σc * εc
                Ψ_eff = 2 * material.E * (ε_max*(1+material.l_r)/2.0)^2 * ((1 - material.l_r) / 2.0)^material.n
                Δα = Ψ_eff 
              end
              α = αₙ + Δα
              #Update state variables
              H = max(Ψ,Hₙ)
              if store
                state[q_point]=MaterialState(H,ϕ,α,Ψ,Ψ_eff)
              end
    
            #Phase field degradation based on old phi
            gdn= (1.0-ϕ)^2 +1.e-7
    
            ##ASSEMBLE DISPLACEMENT PROBLEM
            #Loop over displacement test functions
            for i in 1:nbase_u
    
    
                if material.dim == 2
                    δε_2d = shape_symmetric_gradient(cellvalues_u,q_point,i)
                    δε = SymmetricTensor{2,3,Float64}((δε_2d[1,1],δε_2d[1,2], 0.0,
                        δε_2d[2,2], 0.0, 0.0))
                else
                    δε =shape_symmetric_gradient(cellvalues_u,q_point,i)
                end
                #Add contribution to element rhs
    
                fe[i] += gdn*(δε ⊡ σ)*dΩᵤ
            end
        end
        if material.load == CLA_F
        ∂Ωₜ = cell.grid.facetsets["TOP"] 
          ## Apply traction loading                    
          for facet in 1:nfacets(cell)
              if (cellid(cell), facet) ∈ ∂Ωₜ
                reinit!(facevalues, cell, facet)
                  for q_point in 1:getnquadpoints(facevalues)
                    dΓ = getdetJdV(facevalues, q_point)
                    n = getnormal(facevalues, q_point)
                    trac_n = (trac ⋅ n) * n 
                    for i in 1:nbase_u
                        δu = shape_value(facevalues, q_point, i)
                        fe[i] -= δu ⋅ trac_n * dΓ
                    end
                  end
              end
          end
        end
        return fe
end

function assemble_element_phi!(Ke::Matrix, fe::Vector, cellvalues_ϕ::CellValues,
                                ϕe::Vector, material::Brittle, state, state_old)
    nbase_ϕ = getnbasefunctions(cellvalues_ϕ)

    Gc = material.Gc
    ℓ = material.ℓ

    # Loop over integration points
    for q_point in 1:getnquadpoints(cellvalues_ϕ)
        dΩᵩ =getdetJdV(cellvalues_ϕ,q_point)
        #Phase field 
        ϕ = function_value(cellvalues_ϕ,q_point,ϕe)
        ∇ϕ = function_gradient(cellvalues_ϕ,q_point,ϕe)

        α=state[q_point].α
        H=state[q_point].H
        ψ=state[q_point].ψ
        Ψ_eff=state[q_point].Ψ_eff
        #Recover state vairables
        Hₙ = state_old[q_point].H
        ϕₙ = state_old[q_point].ϕ
        ψₙ =state_old[q_point].ψ
        #Phase field irreversibillity condition
        H=state[q_point].H

        #Update state variables
        state[q_point]=MaterialState(H,ϕ,α,ψ,Ψ_eff)
        #fatigue degradation function
        fdeg = fatigue_degradation(α, material)
        #derivative of phase field degradation function 
        gd′ =  -2.0(1.0-ϕ)
        ##ASSEMBLE PHASE FIELD PROBLEM
        for i in 1:nbase_ϕ
            δϕ = shape_value(cellvalues_ϕ,q_point,i)
            δ∇ϕ = shape_gradient(cellvalues_ϕ,q_point,i)
            fe[i] += (gd′*H*δϕ+fdeg*Gc/ℓ *δϕ * ϕ + fdeg*Gc*ℓ*δ∇ϕ ⋅ ∇ϕ)*dΩᵩ
            for j in 1:i
                ϕ′ = shape_value(cellvalues_ϕ,q_point,j)
                ∇ϕ′= shape_gradient(cellvalues_ϕ,q_point,j)
                gd′′ = 2.0ϕ′
                Ke[i,j] += (gd′′*H*δϕ+ fdeg*Gc/ℓ *δϕ*ϕ′+fdeg*Gc*ℓ * δ∇ϕ⋅∇ϕ′)*dΩᵩ
            end
        end
    end
    symmetrize_lower!(Ke)
    return Ke, fe
end

function assemble_residual_phi!(fe::Vector, cellvalues_ϕ::CellValues,
                                ϕe::Vector, material::Brittle, state, state_old, store::Bool)
    nbase_ϕ = getnbasefunctions(cellvalues_ϕ)

    Gc = material.Gc
    ℓ = material.ℓ

    # Loop over integration points
    for q_point in 1:getnquadpoints(cellvalues_ϕ)
        dΩᵩ =getdetJdV(cellvalues_ϕ,q_point)
        ϕ = function_value(cellvalues_ϕ,q_point,ϕe)
        ∇ϕ = function_gradient(cellvalues_ϕ,q_point,ϕe)

        α=state[q_point].α
        ψ=state[q_point].ψ
        Ψ_eff=state[q_point].Ψ_eff
#
        Hₙ = state_old[q_point].H
        ϕₙ = state_old[q_point].ϕ
        ψₙ =state_old[q_point].ψ
#
        H=state[q_point].H
#
        if store
            state[q_point]=MaterialState(H,ϕ,α,ψ,Ψ_eff)
        end
#
        fdeg = fatigue_degradation(α, material)
#
        gd′ =  -2.0(1.0-ϕ)
#
        for i in 1:nbase_ϕ
            δϕ = shape_value(cellvalues_ϕ,q_point,i)
            δ∇ϕ = shape_gradient(cellvalues_ϕ,q_point,i)
            fe[i] += (gd′*H*δϕ+fdeg*Gc/ℓ *δϕ * ϕ + fdeg*Gc*ℓ*δ∇ϕ ⋅ ∇ϕ)*dΩᵩ
        end
    end
    return fe
end
function symmetrize_lower!(K)
    for i in 1:size(K,1)
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end;

function assemble_global(q::Vector, cellvalues, facevalues,
                           K::SparseMatrixCSC, dh::DofHandler,
                           material::Brittle, states, states_old, timestep, trac::Vec{2, Float64})
    #allocate element stiffness and rhs
    nbase = getnbasefunctions(cellvalues)

    Ke=zeros(nbase,nbase)
    fe = zeros(nbase)
    f=zeros(ndofs(dh))
    assembler = start_assemble(K,f)
#
    fielddim=nbase
    for (i, cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues,cell)
        fill!(Ke,0)
        fill!(fe,0)
        eldofs=celldofs(cell)
        qe=q[eldofs]
        state = @view states[:, i]
        state_old = @view states_old[:, i]
        if fielddim>4
            assemble_element_u!(Ke,fe,cell, cellvalues,facevalues,qe, material,state,state_old,timestep,trac)
        else
            assemble_element_phi!(Ke,fe,cellvalues,qe, material,state,state_old)
        end
        assemble!(assembler,eldofs,Ke,fe)
    end
    return K,f
end

function assemble_global_r(q::Vector, cellvalues, facevalues, dh::DofHandler,
                           material::Brittle, states, states_old, trac, store::Bool=true)
    #allocate element stiffness and rhs
    nbase = getnbasefunctions(cellvalues)
    fe = zeros(nbase)
    f=zeros(ndofs(dh))
#
    fielddim=nbase
    for (i, cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues,cell)
        fill!(fe,0)
        eldofs=celldofs(cell)
        qe=q[eldofs]
        state = @view states[:, i]
        state_old = @view states_old[:, i]
        if fielddim>4
            assemble_residual_u!(fe,cell,cellvalues, facevalues, qe, material,state,state_old,trac,store)
        else
            assemble_residual_phi!(fe,cellvalues,qe, material,state,state_old,store)
        end
        f[eldofs] += fe
    end
    return f
end

# Solver functions
function Newton_raphson!(q::Vector, K::SparseMatrixCSC, cellvalues, facevalues, dh::DofHandler, ch::ConstraintHandler,
                        grid::Grid, Material::Brittle, states, states_old, timestep, trac, tag::String,  solver::SolverState)
    K_fac = nothing
    iterations = 0
    norm_r = Inf  # Initialize norm_r
    
    try
        for nitr = 1:(solver.nᵢ+1)
            if nitr > solver.nᵢ
                println("\nWARNING: Newton-Raphson failed to converge in $(solver.nᵢ) iterations")
                println("Final residual norm: $norm_r")
                return K_fac, q, iterations, norm_r  # Return instead of error
            end
            
            K, r = assemble_global(q, cellvalues, facevalues, K, dh, Material, states, states_old, timestep, trac)
            apply_zero!(K,r,ch)
            norm_r = maximum(abs.(r[Ferrite.free_dofs(ch)]))
            
            if (norm_r < solver.NewtonTOL) && (nitr > 1)
                break
            end
            
            iterations += 1
            if isposdef(K)
                K_fac = cholesky(K)
            else
                println("Matrix not positive definite for $tag, using LU factorization")
                K_fac = lu(K)
            end
            
            try
                Δq = K_fac\r
                q -= Δq
            catch e
                println("\nWARNING: Linear solver failed in iteration $nitr")
                println("Error: ", e)
                return K_fac, q, iterations, norm_r
            end
        end
        
        return K_fac, q, iterations, norm_r
        
    catch e
        println("\nERROR in Newton-Raphson solver:")
        println(e)
        return K_fac, q, iterations, norm_r
    end
end

function Mod_Newton_Raphson!(q::Vector, K, cellvalues, facevalues, dh::DofHandler, ch::ConstraintHandler,
                        grid::Grid, Material::Brittle, states, states_old, trac, tag::String,  solver::SolverState)
    iterations = 0
    qold = deepcopy(q)
    fail=false
    norm_r = Inf  # Initialize norm_r here
    for nitr = 1:(solver.nᵢ+1)
        if nitr > solver.nᵢ
            fail=true
            q=qold
            break
        end
        r = assemble_global_r(q, cellvalues, facevalues, dh, Material,states,states_old,trac);
        apply_zero!(r,ch)
        norm_r = maximum(abs.(r[Ferrite.free_dofs(ch)]))
        if (norm_r < solver.NewtonTOL) && (nitr > 1)
            break
        end
        iterations += 1
        Δq = K\r
        q -= Δq
    end
    #    print(tag*" converged in $iterations iterations \n")
    return q, iterations, fail, norm_r
end

function DetermineSolver!(solver::SolverState, no_restart::Int, dΨ_eff::Float64, output::OutputVariables)
    Ψ_eff_jump = dΨ_eff > solver.dΨ_tol
    total_iterations = output.totalIterations_phi + output.totalIterations_u
    high_iterations = (total_iterations - output.totalIterations_outer) > 8  
    
    if solver.rebuild_start
        println("REBUILDING STIFFNESS (rebuild_start flag triggered)")
        solver.strategy = NewtonRaphson
        solver.rebuild_counter += 1
        solver.no_restart = 0
        solver.rebuild_start = false
    elseif solver.rebuild_counter > 3
        println("REBUILDING STIFFNESS (rebuild_counter > 3)")
        solver.strategy = NewtonRaphson
        solver.rebuild_counter += 1
        solver.no_restart = 0
        solver.rebuild_start = true
    elseif CONFIG.solverscheme == Adaptive
        if Ψ_eff_jump || high_iterations
            # Calculate minimum allowed n_c (not less than 2)
            if high_iterations  
                new_nc = max(2, solver.base_n_c ÷ 4)  
                println("HIGH ITERATIONS DETECTED: Iterations > 8")
            else  # Energy jump only
                new_nc = max(2, solver.base_n_c ÷ 2)  
                println("ENERGY JUMP DETECTED: $dΨ_eff > $(solver.dΨ_tol)")
            end
#            
            if new_nc < solver.current_n_c  # Only update if it would reduce n_c
                solver.current_n_c = new_nc
                solver.adaptive_mode = true
                println("Adapting: n_c reduced to $(solver.current_n_c)")
            end
        elseif !Ψ_eff_jump && !high_iterations && solver.adaptive_mode && dΨ_eff < solver.dΨ_tol/2
            # When conditions improve, restore original n_c
            solver.current_n_c = solver.base_n_c
            solver.adaptive_mode = false
            println("Exiting adaptive mode: n_c restored to $(solver.base_n_c)")
        end

        # Periodic switch to NR based on current_n_c
        if mod(solver.no_restart, solver.current_n_c) == 0
            println("PERIODIC REBUILD (n_c = $(solver.current_n_c))")
            solver.strategy = NewtonRaphson
            solver.rebuild_counter += 1
            solver.no_restart = 0
        else
            solver.strategy = ModifiedNewtonRaphson
        end
    else  # Standard mode (adaptive switching scheme not activated)
        if mod(solver.no_restart, solver.base_n_c) == 0
            println("PERIODIC REBUILD (n_c = $(solver.base_n_c))")
            solver.strategy = NewtonRaphson
            solver.rebuild_counter += 1
            solver.no_restart = 0
        else
            solver.strategy = ModifiedNewtonRaphson
        end
    end
    
    output.totalIterations_outer = total_iterations
    
    if solver.no_restart == 0
        println("Ψ_eff change between timesteps: $dΨ_eff")
        println("Latest step iterations: $(total_iterations - output.totalIterations_outer)")
        println("Current n_c: $(solver.current_n_c), Adaptive mode: $(solver.adaptive_mode)")
        println("Update strategy: $(solver.strategy)")
    end
end

function OutputForce(q::Vector, cellvalues, facevalues, dh::DofHandler,
    grid::Grid, Material::Brittle, states, states_old, set::String)
    trac = Vec{2, Float64}((0.0, 0.0))
    F  =assemble_global_r(q, cellvalues,facevalues, dh, Material,states,states_old,trac, false);
    F_x = 0.0
    F_y = 0.0
    
    if set ∈ keys(grid.nodesets)
        outputset = grid.nodesets[set]
    elseif set ∈ keys(grid.facesets)
        println("facesets are currently not supported for force output")
        return 0.0, 0.0
    else
        println("Warning: invalid set for force output")
        return 0.0, 0.0
    end
    
    for node in outputset
        dofs = Ferrite.dofs_for_node(dh, node)
        if length(dofs) >= 2
            F_x += F[dofs[1]]
            F_y += F[dofs[2]]
        end
    end
    
    return F_x, F_y
end

# Main problem function
function Problem(solver::SolverState, output::OutputVariables, cellvalues_ϕ::CellValues, cellvalues_u::CellValues, 
                 facevalues::FacetValues, dh_phi::DofHandler, dh_u::DofHandler, ch_phi::ConstraintHandler, ch_u::ConstraintHandler,
                 grid::Grid, Material::Brittle, n_timesteps::Int)

    CrackTol = 0.95

    #Create sparsity patterns
    u = zeros(ndofs(dh_u))
    ϕ  =zeros(ndofs(dh_phi))
    K_u=allocate_matrix(dh_u)
    K_ϕ=allocate_matrix(dh_phi)

    #Initialize space for factorized matrices (almost certainly not the best way to do this)
    K_ϕ_fac= Cholesky
    K_u_fac = Cholesky

    # Create material states. One array for each cell, where each element is an array of material-
    # states - one for each integration point
    nqp = getnquadpoints(cellvalues_u)
    states = [MaterialState() for _ in 1:nqp, _ in 1:getncells(grid)]
    states_old = [MaterialState() for _ in 1:nqp, _ in 1:getncells(grid)]
    results = DataFrame(
        Timestep = Int[],
        CrackExtent = Float64[],
        Alpha = Float64[],
        Hist = Float64[],
        fdeg = Float64[],
        Ψ_eff = Float64[]
    )
    start_time = time()  # Record start time
    trac = Vec{2}((0.0, 0.0))
    Disp = 0

    #begin computation
    for timestep in 0:n_timesteps

        if solver.loading == Monotonic
            Disp= timestep*solver.u_max/n_timesteps
        elseif solver.loading == Fatigue
            Disp = sin(timestep*3.141592/2.)*solver.u_max
        elseif solver.loading == CLA_U   # Constant load amplitude for displacement controlled case
            Disp = timestep== 0 ? 0 : solver.u_max
        elseif solver.loading == CLA_F   # Constant load amplitude for force controlled case
            traction_magnitude = timestep == 0 ? 0 : solver.P_max 
            trac = Vec{2}((0.0, traction_magnitude))
        end  
  
#        update!(ch_u, Disp) # evaluates the D-bndc at time t
        update!(ch_u, Disp) # evaluates the D-bndc at time t
        update!(ch_phi)
        apply!(u, ch_u)  # set the prescribed values in the solution vector
        apply!(ϕ, ch_phi)  # set the prescribed values in the solution vector
        solver.rebuild_counter= 0

        #OUTER NR LOOP
        dΨ_eff = solver.Ψ_eff_new - solver.Ψ_eff_old
        
        DetermineSolver!(solver, solver.no_restart, dΨ_eff, output)  

        if solver.strategy == ModifiedNewtonRaphson
            # Solve phase field (φ)
            ϕ, nitr_phi, fail, norm_r = Mod_Newton_Raphson!(ϕ, K_ϕ_fac, cellvalues_ϕ, facevalues, 
                                    dh_phi, ch_phi, grid, Material, states, states_old, trac, "ϕ", solver)
            output.totalIterations_phi += nitr_phi
            
            if !fail
                # Solve displacement field (u)
                u, nitr_u, fail, norm_r = Mod_Newton_Raphson!(u, K_u_fac, cellvalues_u, facevalues,
                                    dh_u, ch_u, grid, Material, states, states_old, trac, "u", solver)
                output.totalIterations_u += nitr_u
            end
        
            if !fail
                print("\n Time step @time = $timestep completed. Phase field iterations: $nitr_phi, Displacement iterations: $nitr_u\n")
            end
        
            if fail
                print("Backtracking!")
                solver.strategy = NewtonRaphson
            end
        end
        
        if solver.strategy == NewtonRaphson
            print("\n REBUILDING STIFFNESS at time = $timestep")
            # Solve with rebuilt stiffness matrices
            K_ϕ_fac, ϕ, nitr_phi, norm_r = Newton_raphson!(ϕ, K_ϕ, cellvalues_ϕ, facevalues,
                                    dh_phi, ch_phi, grid, Material, states, states_old, timestep, trac, "ϕ", solver)
            output.totalIterations_phi += nitr_phi
            output.matrixFactorizations_phi += nitr_phi
        
            K_u_fac, u, nitr_u, norm_r = Newton_raphson!(u, K_u, cellvalues_u, facevalues,
                                    dh_u, ch_u, grid, Material, states, states_old, timestep, trac, "u", solver)
            output.totalIterations_u += nitr_u
            output.matrixFactorizations_u += nitr_u
        
            print(" completed. Phase field iterations: $nitr_phi, Displacement iterations: $nitr_u\n")
        
            solver.no_restart = 0
            solver.rebuild_counter += 1
        end
        
        # Update counters and flags
        solver.no_restart += 1
        if solver.no_restart == solver.rebuild_frequency_min
            solver.rebuild_start = true 
        end
        #Write output for contour plots
        if mod(timestep,output.plotFrequency)==0 || timestep==n_timesteps
            simtime=timestep/n_timesteps
            pvd = WriteVTK.paraview_collection("TimeSeries", append=true)
            VTKGridFile("Contour_$(output.plotframe)", dh_phi) do vtk
                write_solution(vtk, dh_phi, ϕ)
                write_solution(vtk, dh_u, u)
                pvd[simtime] = vtk
            end
            WriteVTK.vtk_save(pvd)
            output.plotframe += 1
        end
        #Write history output
        if mod(timestep,output.historyFrequency)==0 || timestep==n_timesteps
            #Crack Extension
            CrackExtent= CrackTrack(ϕ, dh_phi,cellvalues_ϕ,grid,output.a0,output.CrackDirection)
            solver.Ψ_eff_old = solver.Ψ_eff_new
            solver.Ψ_eff_new = maximum(state.Ψ_eff for state in states)
            H = maximum(state.H for state in states)
            Ψ_eff = maximum(state.Ψ_eff for state in states)
            α = maximum(state.α for state in states)
            fdeg = fatigue_degradation(α, Material)
            push!(results, [timestep, CrackExtent, α, H, fdeg, Ψ_eff])
                # Write the DataFrame to a CSV file
            CSV.write("output_data.csv", results)

            if CrackExtent >= CONFIG.max_crack_length
                end_time = time()
                runtime = end_time - start_time
                hours = floor(Int, runtime / 3600)
                minutes = floor(Int, (runtime % 3600) / 60)
                seconds = floor(Int, runtime % 60)
                println("\nSimulation stopped: Crack length ($CrackExtent) reached maximum length ($(CONFIG.max_crack_length))")
                println("Computation complete! Runtime information:")
                println("Total runtime: $hours hours, $minutes minutes, $seconds seconds")
                println("Total matrix factorizations (u): $(output.matrixFactorizations_u)")
                println("Total matrix factorizations (phi): $(output.matrixFactorizations_phi)")
                println("Total iterations (u): $(output.totalIterations_u)")
                println("Total iterations (phi): $(output.totalIterations_phi)")
            end
        end
        states_old .= states
    end
    end_time = time()
    runtime = end_time - start_time
    hours = floor(Int, runtime / 3600)
    minutes = floor(Int, (runtime % 3600) / 60)
    seconds = floor(Int, runtime % 60)
    
    println("Computation complete! Runtime information:")
    println("Total runtime: $hours hours, $minutes minutes, $seconds seconds")
    println("Total matrix factorizations (u): $(output.matrixFactorizations_u)")
    println("Total matrix factorizations (phi): $(output.matrixFactorizations_phi)")
    println("Total iterations (u): $(output.totalIterations_u)")
    println("Total iterations (phi): $(output.totalIterations_phi)")
end

# Code Execution
function main()
    # Define problem domain
    grid = get_ferrite_grid("sent.inp")
    dim = 2

    ip_u = Lagrange{RefQuadrilateral, 1}()
    ip_phi = Lagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    face_qr = FacetQuadratureRule{RefQuadrilateral}(2)

    cellvalues_u = CellValues(qr, ip_u^dim)
    cellvalues_ϕ = CellValues(qr, ip_phi)
    facevalues = FacetValues(face_qr, ip_u^dim)
    
    l_r = CONFIG.u_min / CONFIG.u_max
    P_max, P_min = CONFIG.F_max/CONFIG.area, CONFIG.F_min/CONFIG.area   #for the force-controlled problem
    # Set up constraint handlers
    ∂Ω₁ = create_facetset(grid, getnodeset(grid,"TOP"))
    ∂Ω₂ = create_facetset(grid, getnodeset(grid,"BTM"))
    dbc₁ = Dirichlet(:u, ∂Ω₁, (x,t) -> t, 2)
    dbc₂ = Dirichlet(:u, ∂Ω₂, (x,t) -> [0, 0], [1, 2])

    # Set up dofHandlers
    dh_u = DofHandler(grid)
    add!(dh_u, :u, ip_u^dim)
    close!(dh_u)

    dh_phi = DofHandler(grid)
    add!(dh_phi, :ϕ, ip_phi)
    close!(dh_phi)

    # Define constraint handlers
    ch_u = ConstraintHandler(dh_u)
    add!(ch_u, dbc₁)
    add!(ch_u, dbc₂)
    close!(ch_u)
    update!(ch_u, 0.0)
    ch_phi = ConstraintHandler(dh_phi)
    close!(ch_phi)
    update!(ch_phi, 0.0)

    loadingtype = CONFIG.loadingtype 
    dmode = CONFIG.dmode
    smode = CONFIG.smode

    Material = Brittle(CONFIG.E, CONFIG.ν, CONFIG.Gc, CONFIG.ℓ, dmode, CONFIG.loadingtype, 
                       dim, l_r, CONFIG.fmode, CONFIG.n, CONFIG.κ, CONFIG.αₜ, smode)

    output = OutputVariables(CONFIG.plot_frequency, CONFIG.history_output_frequency, CONFIG.a₀, CONFIG.CrackDirection, CONFIG.outputset)

    solver = SolverState(CONFIG.rebuild_frequency_base, CONFIG.n_c, P_max, P_min, CONFIG.u_max, CONFIG.u_min, CONFIG.area, CONFIG.thck, CONFIG.n_timesteps, CONFIG.nᵢ, 1.e-5, CONFIG.dΨ_tol, loadingtype)

    # Start solving problem
    Problem(solver, output, cellvalues_ϕ, cellvalues_u, facevalues, dh_phi, dh_u,
                                    ch_phi, ch_u, grid, Material, CONFIG.n_timesteps)
end

# Run the main function
main()
