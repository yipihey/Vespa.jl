using Printf

const BOX = 1.0
const X = [3.5, 10.5, 6.5, 12.5] ./ 16.0
const Y = [5.5, 5.5, 11.5, 9.5] ./ 16.0
const Z = [7.5, 7.5, 4.5, 13.5] ./ 16.0
const M = ones(Float64, 4)

function periodic_image_sum_accel(x, y, z, m; boxsize = 1.0, nimg = 1)
    n = length(x)
    ax = zeros(Float64, n)
    ay = zeros(Float64, n)
    az = zeros(Float64, n)
    @inbounds for i in 1:n
        xi = x[i]
        yi = y[i]
        zi = z[i]
        for j in 1:n
            mj = m[j]
            for nx in -nimg:nimg
                for ny in -nimg:nimg
                    for nz in -nimg:nimg
                        if i == j && nx == 0 && ny == 0 && nz == 0
                            continue
                        end
                        dx = (x[j] - xi) + nx * boxsize
                        dy = (y[j] - yi) + ny * boxsize
                        dz = (z[j] - zi) + nz * boxsize
                        r2 = dx * dx + dy * dy + dz * dz
                        invr = inv(sqrt(r2))
                        invr3 = invr * invr * invr
                        coeff = mj * invr3
                        ax[i] += coeff * dx
                        ay[i] += coeff * dy
                        az[i] += coeff * dz
                    end
                end
            end
        end
    end
    return ax, ay, az
end

function momentum_residual(m, ax, ay, az)
    return sum(m .* ax), sum(m .* ay), sum(m .* az)
end

function main()
    println("AREPO periodic direct convention smoke")
    println("fixture: 4 equal-mass particles in a unit periodic box")
    println("method: symmetric finite image-sum diagnostic")
    println("note: this is a diagnostic scaffold, not the recommended PM gate convention")
    println()
    println(" nimg | max|a|         | sum(m*ax)       sum(m*ay)       sum(m*az)       | max residual")
    println("----- | -------------- | --------------  --------------  --------------  | ------------")
    for nimg in 0:2
        ax, ay, az = periodic_image_sum_accel(X, Y, Z, M; boxsize = BOX, nimg = nimg)
        residual = momentum_residual(M, ax, ay, az)
        maxa = maximum(sqrt.(ax .* ax .+ ay .* ay .+ az .* az))
        rmax = maximum(abs, residual)
        @printf("%5d | % .10e | % .10e  % .10e  % .10e | %.3e\n",
                nimg, maxa, residual[1], residual[2], residual[3], rmax)
    end
    println()
    println("recommended convention: force-only background-subtracted periodic force")
end

main()
