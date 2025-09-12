using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using SpecialFunctions
using Test

function r_factor(mu, nu)
    if iseven(mu/2) && iseven(nu/2)
        gamma(mu/4 + 1/2)*gamma(nu/4 + 1/2)/gamma((mu+nu)/4 + 2)
    elseif iseven((mu-1)/2) && iseven((nu-1)/2)
        sqrt(pi)/2*gamma(mu/4 + 1/4)*gamma(nu/4 + 1/4)/gamma((mu+nu)/4 + 2)
    else
        zero(gamma(mu/4 + 1/2))
    end
end

function partial_factorial(start::Integer, stop::Integer)
    if start == stop
        return stop
    elseif start > stop
        return 0
    else
        return prod(start:stop)
    end
end

function _alm_coeff(l, m)
    if m == 0
        sqrt((2*l+1)/(4*pi))
    elseif m > 0
        sqrt(2*(2*l+1)/(4*pi*partial_factorial(l-m+1, l+m)))
    else
        sqrt(2*(2*l+1)*partial_factorial(l+m+1, l-m)/4*pi)
    end
end

function _blmjk_coeff(l, m, j, k)
    try
        (1 << l)*factorial(m)*gamma((l+m+k+1)/2)/(factorial(j)*factorial(k)*factorial(m-j)*factorial(l-m-k)*gamma((-l+m+k+1)/2))
    catch DomainError
        0
    end
end

function _cpqk_coeff(p,q,k)
    factorial(Int(k/2))/(factorial(Int(q/2))*factorial(Int((k-p)/2))*factorial(Int((p-q)/2)))
end

function polynomial_n_to_xyz_exponent(n)
    l = Int(floor(sqrt(n)))
    m = n - l*l - l

    mu = l-m
    nu = l+m

    if nu % 2 == 0
        (Int(mu/2), Int(nu/2), 0)
    else
        (Int((mu-1)/2), Int((nu-1)/2), 1)
    end
end

function xyz_exponent_to_polynomial_n(a, b, c)
    if c == 0
        mu = 2*a
        nu = 2*b
    else
        mu = 2*a + 1
        nu = 2*b + 1
    end

    l = Int((mu+nu)/2)
    m = Int((nu-mu)/2)

    l*l + l + m    
end

function A_matrix(lmax)
    A = zeros((lmax+1)^2, (lmax+1)^2)

    for l in 0:lmax
        for m in -l:l
            if m >= 0
                for j in 0:2:m
                    for k in 0:2:l-m
                        for p in 0:2:k
                            for q in 0:2:p
                                sgn = ((j+p)/2 % 2 == 0) ? 1 : -1
                                coeff = sgn*_alm_coeff(l, m)*_blmjk_coeff(l, m, j, k)*_cpqk_coeff(p, q, k)
                                n = xyz_exponent_to_polynomial_n(m-j+p-q, j+q, 0)
                                A[n+1, l*l + l + m + 1] += coeff
                            end
                        end
                    end
                end

                for j in 0:2:m
                    for k in 1:2:l-m
                        for p in 0:2:k-1
                            for q in 0:2:p
                                sgn = ((j+p)/2 % 2 == 0) ? 1 : -1
                                coeff = sgn*_alm_coeff(l, m)*_blmjk_coeff(l, m, j, k)*_cpqk_coeff(p, q, k-1)
                                n = xyz_exponent_to_polynomial_n(m-j+p-q, j+q, 1)
                                A[n+1, l*l + l + m + 1] += coeff
                            end
                        end
                    end
                end
            else
                for j in 1:2:abs(m)
                    for k in 0:2:l-abs(m)
                        for p in 0:2:k
                            for q in 0:2:p
                                sgn = ((j+p-1)/2 % 2 == 0) ? 1 : -1
                                coeff = sgn*_alm_coeff(l, abs(m))*_blmjk_coeff(l, abs(m), j, k)*_cpqk_coeff(p, q, k)
                                n = xyz_exponent_to_polynomial_n(abs(m)-j+p-q, j+q, 0)
                                A[n+1, l*l + l + m + 1] += coeff
                            end
                        end
                    end
                end

                for j in 1:2:abs(m)
                    for k in 1:2:l-abs(m)
                        for p in 0:2:k-1
                            for q in 0:2:p
                                sgn = ((j+p-1)/2 % 2 == 0) ? 1 : -1
                                coeff = sgn*_alm_coeff(l, abs(m))*_blmjk_coeff(l, abs(m), j, k)*_cpqk_coeff(p, q, k-1)
                                n = xyz_exponent_to_polynomial_n(abs(m)-j+p-q, j+q, 1)
                                A[n+1, l*l + l + m + 1] += coeff
                            end
                        end
                    end
                end
            end
        end
    end
    A
end

# Rotations, from Ivanic & Ruedenberg, J. Phys. Chem., 100, 6342 (1996).
function _u(l,m,mp)
    if abs(mp) < l
        sqrt((l+m)*(l-m)/((l+mp)*(l-mp)))
    else
        sqrt((l+m)*(l-m)/((2*l)*(2*l-1)))
    end
end

function _v(l,m,mp)
    if abs(mp) < l
        c = m==0 ? -sqrt(2) : 1
        c/2*sqrt((l+abs(m)-1)*(l+abs(m))/((l+mp)*(l-mp)))
    else
        c = m==0 ? -sqrt(2) : 1
        c/2*sqrt((l+abs(m)-1)*(l+abs(m))/((2*l)*(2*l-1)))
    end
end

function _w(l,m,mp)
    if abs(mp) < l
        c = m==0 ? 0 : 1
        -c/2*sqrt((l-abs(m)-1)*(l-abs(m))/((l+mp)*(l-mp)))
    else
        c = m==0 ? 0 : 1
        -c/2*sqrt((l-abs(m)-1)*(l-abs(m))/((2*l)*(2*l-1)))
    end
end

function _i(l,m)
    m + l + 1
end

function _P(l,i,m,mp,R,Rm1)
    if abs(mp) < l
        R[_i(1,i),_i(1,0)]*Rm1[_i(l-1,m),_i(l-1,mp)]
    elseif mp == l
        R[_i(1,i),_i(1,1)]*Rm1[_i(l-1,m),_i(l-1,mp-1)] - R[_i(1,i),_i(1,-1)]*Rm1[_i(l-1,m),_i(l-1,-mp+1)]
    else # mp == -l
        R[_i(1,i),_i(1,1)]*Rm1[_i(l-1,m),_i(l-1,mp+1)] + R[_i(1,i),_i(1,-1)]*Rm1[_i(l-1,m),_i(l-1,-mp-1)]
    end
end

function _U(l,m,mp,R,Rm1)
    _P(l,0,m,mp,R,Rm1)
end

function _V(l,m,mp,R,Rm1)
    if m == 0
        _P(l,1,1,mp,R,Rm1) + _P(l,-1,-1,mp,R,Rm1)
    elseif m > 0
        cp = m==1 ? sqrt(2) : 1
        cm = m==1 ? 0 : 1

        cp*_P(l,1,m-1,mp,R,Rm1) - cm*_P(l,-1,-m+1,mp,R,Rm1)
    else # m < 0
        cm = m==-1 ? 0 : 1
        cp = m==-1 ? sqrt(2) : 1

        cm*_P(l,1,m+1,mp,R,Rm1) + cp*_P(l,-1,-m-1,mp,R,Rm1)
    end
end

function _W(l,m,mp,R,Rm1)
    if m >= 0 # m==0 never occurs because of test _w == 0 below
        _P(l,1,m+1,mp,R,Rm1) + _P(l,-1,-m-1,mp,R,Rm1)
    else
        _P(l,1,m-1,mp,R,Rm1) - _P(l,-1,-m+1,mp,R,Rm1)
    end
end

function _R(l,m,mp,R,Rm1)
    u = _u(l,m,mp)
    v = _v(l,m,mp)
    w = _w(l,m,mp)

    (u==0 ? 0 : u*_U(l,m,mp,R,Rm1)) + (v==0 ? 0 : v*_V(l,m,mp,R,Rm1)) + (w==0 ? 0 : w*_W(l,m,mp,R,Rm1))
end

function _next_R(l, R, Rm1)
    [_R(l, m, mp, R, Rm1) for m in -l:l, mp in -l:l]
end

function _R1(alpha, beta, gamma)
    ca = cos(alpha)
    sa = sin(alpha)

    cb = cos(beta)
    sb = sin(beta)

    cg = cos(gamma)
    sg = sin(gamma)

    [(ca*cg - cb*sa*sg) (sb*sg) (-cg*sa - ca*cb*sg)
     (sa*sb) (cb) (ca*sb)
     (ca*sg + cb*cg*sa) (-cg*sb) (ca*cb*cg - sa*sg)]
end

function R_matrices(lmax, alpha, beta, gamma)
    R = _R1(alpha, beta, gamma)
    Rs = (R,)
    for l in 2:lmax
        RR = _next_R(l, R, Rs[end])
        Rs = (Rs..., RR)
    end
    Rs
end

function rotation_test(lmax)
    a = rand(0:2*pi)
    b = rand(0:2*pi)
    g = rand(0:2*pi)

    Rs = R_matrices(lmax, a, b, g)

    for (l, R) in zip(1:lmax, Rs)
        @test size(R) == (2l+1, 2l+1)
        @test R'*R ≈ Matrix(I, 2l+1, 2l+1)
    end
end