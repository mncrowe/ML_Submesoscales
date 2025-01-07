# The parameters in this example correspond roughly to those used by [taylor2016](@citet) and lead to the
# generation of a single submesoscale eddy.

# ## Model setup

using Oceananigans, Printf
using Oceananigans.Units
using Oceananigans.Fields
using Random

Random.seed!(11)

Nx = 512
Ny = 512
Nz = 64

#grid = RectilinearGrid(GPU(), size = (Nx, Ny, Nz), extent = (20kilometer, 20kilometer, 100meters))	# large domain, jet forms
grid = RectilinearGrid(GPU(), size = (Nx, Ny, Nz), extent = (10kilometer, 10kilometer, 100meters))	# small domain, no jet
println(grid)

# coriolis parameter
coriolis = FPlane(f = 1e-4) # [s⁻¹]

# Specify parameters that are used to construct the background state.
background_state_parameters = ( M = 1e-4,       # s⁻¹, geostrophic shear
                                f = coriolis.f, # s⁻¹, Coriolis parameter
                                N = 1e-4,       # s⁻¹, buoyancy frequency
                                H = grid.Lz )

B(x, y, z, t, p) = p.M^2 * x + p.N^2 * (z + p.H)
V(x, y, z, t, p) = p.M^2 / p.f * (z + p.H)

V_field = BackgroundField(V, parameters = background_state_parameters)
B_field = BackgroundField(B, parameters = background_state_parameters)

# Specify some horizontal and vertical viscosity and diffusivity.
νᵥ = κᵥ = 1e-4 # [m² s⁻¹]
vertical_diffusivity = VerticalScalarDiffusivity(ν = νᵥ, κ = κᵥ)

# Model instantiation
model = NonhydrostaticModel(; grid,
                              advection = WENO(grid),
                              timestepper = :RungeKutta3,
                              coriolis,
                              tracers = :b,
                              buoyancy = BuoyancyTracer(),
                              background_fields = (b = B_field, v = V_field),
                              closure = vertical_diffusivity)

# IC, start with a bit of random noise added to the background thermal wind 
Ξ(z) = randn() * z / grid.Lz * (z / grid.Lz + 1)
Ũ = 1e-3
uᵢ(x, y, z) = Ũ * Ξ(z)
vᵢ(x, y, z) = Ũ * Ξ(z)

set!(model, u=uᵢ, v=vᵢ)

simulation = Simulation(model, Δt = 1minutes, stop_time = 200days)

# Adapt the time step while keeping the CFL number fixed.
wizard = TimeStepWizard(cfl = 0.75, diffusive_cfl = 0.75, max_Δt = 60minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))
nothing

# Create a progress message.
progress(sim) = @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.run_wall_time),
                        prettytime(sim.Δt),
                        AdvectiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

# Here, we add some diagnostics to calculate and output
u, v, w = model.velocities
pHy,PNHy = model.pressures
b = model.tracers.b

zeta = Field(∂x(v) - ∂y(u))
KE   = Field((u^2 + v^2 + w^2)/2)
p    = Field(pHy + PNHy)
M4   = Field(∂x(b)^2 + ∂y(b)^2)		# horizontal buoyancy gradient (squared)
#diss = Field(∂x(u)^2 + ∂x(v)^2 + ∂x(w)^2 + ∂y(u)^2 + ∂y(v)^2 + ∂y(w)^2 + ∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2)

b_tot = Average(b, dims=(1, 2))

w_avg  = Average(w, dims=(3))	# depth-avg vertical velocity
b_avg  = Average(b, dims=(3))	# depth-avg buoyancy
bw_avg = Average(b*w, dims=(3))	# vertical buoyancy flux
w_rms  = Average(w^2, dims=(3))	# RMS vertical velocity

simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; u, v, w, b, zeta),
                                                      schedule = TimeInterval(60hours),
                                                      filename = "eady_turbulence_512",
                                                      overwrite_existing = true)

simulation.output_writers[:surface_slice_writer] = NetCDFOutputWriter(model, (; KE, p, b, zeta, M4),
                                                      schedule = TimeInterval(1hours),
                                                      filename = "eady_turbulence_512_surf",
                                                      overwrite_existing = true,
                                                      indices=(:,:,grid.Nz))

simulation.output_writers[:averages] = NetCDFOutputWriter(model, (; bw_avg, w_avg, b_avg, b_tot, w_rms),
                                                      schedule = TimeInterval(1hours),
                                                      #schedule = AveragedTimeInterval(1hours,window = 1hours),	# time averaging
                                                      filename = "eady_turbulence_512_avg",
                                                      overwrite_existing = true)

run!(simulation)
