# Piecewise-linear (MUSCL) reconstruction with the minmod limiter (kernel, P1).
# Second-order in smooth regions, TVD across discontinuities. Operates per
# primitive component; needs one neighbor on each side (so nghost ≥ 2 for the
# boundary faces).

@inline function minmod(a::T, b::T) where {T}
    if a * b <= 0
        return zero(T)
    else
        return abs(a) < abs(b) ? a : b
    end
end

"Limited slope of a primitive component at a cell, from its neighbor differences."
@inline limited_slope(left::T, center::T, right::T) where {T} =
    minmod(center - left, right - center)
