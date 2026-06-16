"""
    ArepoCosmologyStepMetadata

Small typed summary of AREPO-style flat matter + Lambda cosmology step
coefficients.  `expansion_factor` stores the dimensionless `E(a)` term and
`hubble_a` stores the AREPO-style `H(a)` coefficient used by future KDK loops.
"""
struct ArepoCosmologyStepMetadata
    runtime::ArepoCosmologyRuntime
    scale_factor::Float64
    expansion_factor::Float64
    hubble_a::Float64
end

@inline function _arepo_cosmology_require(enabled::Bool, value, name::Symbol)
    enabled && value === nothing &&
        error("arepo_cosmology_step_metadata: $(name) is required when ComovingIntegrationOn=1")
    return value
end

@inline function _arepo_cosmology_flat_lambda_factor(omega0::Real, omega_lambda::Real,
                                                     a::Real)
    a > 0 || error("arepo_cosmology_step_metadata: scale factor must be positive")
    e2 = float(omega0) / float(a)^3 + float(omega_lambda)
    e2 >= 0 || error("arepo_cosmology_step_metadata: Omega0 / a^3 + OmegaLambda must be nonnegative")
    return sqrt(e2)
end

"""
    arepo_cosmology_step_metadata(runtime; a)

Return a compact, dependency-free summary of the cosmology step coefficients
for a flat matter + Lambda background.  When comoving integration is disabled,
the helper returns identity coefficients and leaves `hubble_a` at zero.
"""
function arepo_cosmology_step_metadata(runtime::ArepoCosmologyRuntime; a::Real)
    scale_factor = float(a)
    if !runtime.enabled
        return ArepoCosmologyStepMetadata(runtime, scale_factor, 1.0, 0.0)
    end

    omega0 = _arepo_cosmology_require(true, runtime.omega0, :Omega0)
    omega_lambda = _arepo_cosmology_require(true, runtime.omega_lambda, :OmegaLambda)
    hubble_param = _arepo_cosmology_require(true, runtime.hubble_param, :HubbleParam)
    expansion_factor = _arepo_cosmology_flat_lambda_factor(omega0, omega_lambda, scale_factor)
    hubble_a = float(hubble_param) * expansion_factor
    return ArepoCosmologyStepMetadata(runtime, scale_factor, expansion_factor, hubble_a)
end

arepo_cosmology_step_metadata(params::ArepoParameterSet; a::Real) =
    arepo_cosmology_step_metadata(params.normalized.cosmology; a = a)

arepo_cosmology_step_metadata(normalized::NamedTuple; a::Real) =
    arepo_cosmology_step_metadata(arepo_cosmology_runtime(normalized); a = a)

"""
    arepo_cosmology_expansion_factor(runtime; a)

Return the flat matter + Lambda `E(a)` factor for a given AREPO cosmology
runtime.
"""
arepo_cosmology_expansion_factor(runtime::ArepoCosmologyRuntime; a::Real) =
    arepo_cosmology_step_metadata(runtime; a = a).expansion_factor

arepo_cosmology_expansion_factor(params::ArepoParameterSet; a::Real) =
    arepo_cosmology_step_metadata(params; a = a).expansion_factor

"""
    arepo_cosmology_adot_over_a(runtime; a)

Return the AREPO-style `\\dot a / a` coefficient implied by the current flat
matter + Lambda cosmology parameters.
"""
arepo_cosmology_adot_over_a(runtime::ArepoCosmologyRuntime; a::Real) =
    arepo_cosmology_step_metadata(runtime; a = a).hubble_a

arepo_cosmology_adot_over_a(params::ArepoParameterSet; a::Real) =
    arepo_cosmology_step_metadata(params; a = a).hubble_a
