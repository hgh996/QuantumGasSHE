using StaticArrays
export PTtransform, myint

function PTtransform(ψ::AbstractArray{<:Number})
    ϕ=similar(ψ)
    Nv = round(Int,length(ψ)/2)
    ϕ[1:Nv].= conj.(ψ[Nv+1:end])
    ϕ[Nv+1:end].= conj.(ψ[1:Nv]).*(-1)
    return ϕ
end
##

struct Symc
    n::Int
end

function (C::Symc)(V::Array{Float64,2})
    D = SA[cospi(0.5*C.n) -sinpi(0.5*C.n);sinpi(0.5*C.n) cospi(0.5*C.n)]
    Vq = similar(V)
    @inbounds for iQ in axes(V,2)
        Vq[1,iQ]=D[1,1]*V[1,iQ]+D[1,2]*V[2,iQ]
        Vq[2,iQ]=D[2,1]*V[1,iQ]+D[2,2]*V[2,iQ]
    end
    sz=SA[cispi(-C.n/4) 0.0im;0.0im cispi(C.n/4)]
    return Vq,sz
end

struct Syms
    n::Int
end
function (S::Syms)(V::Array{Float64,2})
    k1=cospi(S.n/4)
    k2=sinpi(S.n/4)
    w1=cospi(1.0)
    Sn=SA[k1^2*(1-w1)+w1 k1*k2*(1-w1);k1*k2*(1-w1) k2^2*(1-w1)+w1]

    Vq=similar(V)
    @inbounds for iQ in axes(V,2)
        Vq[1,iQ]=Sn[1,1]*V[1,iQ]+Sn[1,2]*V[2,iQ]
        Vq[2,iQ]=Sn[2,1]*V[1,iQ]+Sn[2,2]*V[2,iQ]
    end
    sx=SA[0.0im k1-1im*k2;k1+1im*k2 0.0im]
    sz=cis(-0.5pi.*sx)
    return Vq,sz
end

function symPT(V::Array{Float64,2})
    Vq=similar(V)
    @inbounds for iQ in axes(V,2)
        Vq[1,iQ]=-V[1,iQ]
        Vq[2,iQ]=-V[2,iQ]
    end
    sz = SA[0.0 -1.0im;1.0im 0.0].*1.0im
    return Vq,sz
end
function symT(V::Array{Float64,2})
    Vq=copy(V)
    sz = SA[0.0im 1.0;1.0 0.0im]
    return Vq,sz
end

function symTx(V::Array{Float64,2})
    Vq=copy(V)
    # sz=[1.0 00im;0 -1.0]
    sz=SA[-1.0im 0im;0 1.0im]
    return Vq,sz
end

function T_dx!(xx::Array{Float64,3},n::Int)
    Threads.@threads for iy in axes(xx,3)
        @inbounds for ix in axes(xx,2)
            xx[n,ix,iy]+=1pi
        end
    end
end

function TxC2(V::Array{Float64,2})
    D = SA[cospi(0.5*2) -sinpi(0.5*2);sinpi(0.5*2) cospi(0.5*2)]
    Vq = similar(V)
    @inbounds for iQ in axes(V,2)
        Vq[1,iQ]=D[1,1]*V[1,iQ]+D[1,2]*V[2,iQ]
        Vq[2,iQ]=D[2,1]*V[1,iQ]+D[2,2]*V[2,iQ]
    end
    sz=SA[-1.0im 0im;0 1.0im]*SA[cispi(-2/4) 0.0im;0.0im cispi(2/4)]

    return Vq,sz
end

const Tlist=Dict(
    "C1"=>Symc(1), "C2"=>Symc(2), "C3"=>Symc(3), "C4"=>Symc(4),
    "C5"=>Symc(5), "C6"=>Symc(6), "C7"=>Symc(7), "C8"=>Symc(0),
    "S1"=>Syms(1), "S2"=>Syms(2), "S3"=>Syms(3), "S4"=>Syms(4),
    "S5"=>Syms(5), "S6"=>Syms(6), "S7"=>Syms(7), "S8"=>Syms(8),
    "Tx"=>symTx,   "Ty"=>symTx,   "PT"=>symPT,   "T" =>symT, "TxC2"=>TxC2
)

function _Psix(
    v::Vector{ComplexF64},
    Vq::Array{Float64,2},
    sz::AbstractMatrix{ComplexF64},
    xx::Array{Float64,3}
)
    NQ=size(Vq,2)
    _,lx,ly=size(xx)
    
    w=Array{ComplexF64}(undef,2,lx,ly)
    Threads.@threads for iy in 1:ly
        @inbounds for ix in 1:lx
            t1=t2=0.0im
            for iQ in 1:NQ
                tmp=cis(Vq[1,iQ]*xx[1,ix,iy]+Vq[2,iQ]*xx[2,ix,iy])
                t1+=v[iQ]*tmp
                t2+=v[iQ+NQ]*tmp
            end
            w[1,ix,iy]=sz[1,1]*t1+sz[1,2]*t2
            w[2,ix,iy]=sz[2,1]*t1+sz[2,2]*t2
        end
    end
    return w
end

function Psix(v::Vector{ComplexF64},Kvec0::Array{Float64,2},idx::String)
    p0 = SA[0.5pi,0.0]#-0.5.*(a1.+a2)
    a1 = SA[1pi,1pi]
    a2 = SA[-1pi,1pi]
    xx = mymesh([p0, a1, a2],[256,256])

    Kvec,sz = Tlist[idx](Kvec0)
    idx=="Tx" && T_dx!(xx,1)
    idx=="Ty" && T_dx!(xx,2)
    idx=="TxC2" && T_dx!(xx,1)
    
    if idx!="PT" && idx!="T"
        _Psix(v,Kvec,sz,xx)
    else 
        conj.(_Psix(v,Kvec,sz,xx))
    end
end

function _myint(w1::Array{ComplexF64,3},w2::Array{ComplexF64,3})
    _,lx,ly = size(w1)
    tmp = Array{ComplexF64}(undef,lx,ly)
    Threads.@threads for iy in 1:ly
        @inbounds for ix in 1:lx
            tmp[ix,iy]=conj(w1[1,ix,iy])*w2[1,ix,iy]+conj(w1[2,ix,iy])*w2[2,ix,iy]
        end
    end
    trapz((range(0.0,1.0,lx), range(0.0,1.0,ly)),tmp)
end

function myint(
    v1::Vector{ComplexF64},
    v2::Vector{ComplexF64},
    Vq::Array{Float64,2},
    idx::String
)
    w1 = Psix(v1,Vq,"C8")
    w2 = Psix(v2,Vq,idx) 
    _myint(w1,w2)
end