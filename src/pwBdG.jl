export eig_BdG,eigen_BdG,cal_BdG,BdG_bcav

function _intBdgM!(
    matH::Array{ComplexF64,2},
    ψ::Array{ComplexF64,1},
    Kl::Array{Int,2},
    g1::Float64,
    g2::Float64
)
    Nv=length(ψ)
    NQ = round(Int,Nv/2)
    NKl=size(Kl,2)

    @inbounds for ii in 1:NKl
        t1,t2,t3,t4 = view(Kl,:,ii)

        tmp = 2*conj(ψ[t2])*ψ[t3]*g1 + conj(ψ[t2+NQ])*ψ[t3+NQ]*g2
        matH[t1,t4] += tmp
        matH[t1+Nv,t4+Nv] += conj(tmp)

        tmp = conj(ψ[t2+NQ])*ψ[t3]*g2
        matH[t1,t4+NQ] += tmp
        matH[t1+Nv,t4+NQ+Nv] += conj(tmp)

        tmp = conj(ψ[t2])*ψ[t3+NQ]*g2
        matH[t1+NQ,t4] += tmp
        matH[t1+NQ+Nv,t4+Nv] += conj(tmp)
        
        tmp = 2*conj(ψ[t2+NQ])*ψ[t3+NQ]*g1+conj(ψ[t2])*ψ[t3]*g2
        matH[t1+NQ,t4+NQ] += tmp
        matH[t1+NQ+Nv,t4+NQ+Nv] += conj(tmp)
        
        # ---------------- off diag ----------------------
        tmp = ψ[t3]*ψ[t4]*g1
        matH[t1,t2+Nv] += tmp
        matH[t2+Nv,t1] += conj(tmp)

        tmp = ψ[t3+NQ]*ψ[t4]*g2
        matH[t1,t2+NQ+Nv] += tmp
        matH[t2+NQ+Nv,t1] += conj(tmp)

        tmp = ψ[t3]*ψ[t4+NQ]*g2
        matH[t1+NQ,t2+Nv] += tmp
        matH[t2+Nv,t1+NQ] += conj(tmp)

        tmp = ψ[t3+NQ]*ψ[t4+NQ]*g1
        matH[t1+NQ,t2+NQ+Nv] += tmp
        matH[t2+NQ+Nv,t1+NQ] += conj(tmp)
    end
    nothing
end
# BdG 矩阵的相互作用部分，未左乘tz
function intBdgM!(
    matH::Array{ComplexF64,2},
    ev0::Array{ComplexF64,1},
    lat::Lattice
)
    Kl=cal_phi4_coe(lat.Kcoe,lat.Kmax)
    _intBdgM!(matH,ev0,Kl,lat.g1,lat.g2)
    nothing
end

# BdG 矩阵的非对角部分，已左乘tz
function matHu0!(matH::Array{ComplexF64,2},lat::Lattice;pband::Int=0)
    Nm=2*lat.NK
    Mii=diag(matH)

    mat=calmat(lat,[0.0,0.0])
    matH[1:Nm,1:Nm].+=mat
    matH[Nm+1:end,Nm+1:end].+=conj.(mat)

    ben = eigvals(Hermitian(matH),1:20)
    u0::Float64 = ben[1+2*pband] 

    tauz=Diagonal([ones(ComplexF64,Nm); fill(-1.0+0.0im,Nm)])
    lmul!(tauz,matH)

    return Mii,u0
end

# BdG 矩阵的对角部分，已左乘tz
function calmatH!(
    matH::Matrix{ComplexF64},
    Mii::Vector{ComplexF64},
    kk::Vector{Float64},
    Kvec::Array{Float64,2},
    NK::Int,
    mz::Float64,
    u0::Float64
)
    for iQ in 1:NK
        tmp=(kk[1]+Kvec[1,iQ])^2+(kk[2]+Kvec[2,iQ])^2+u0
        matH[iQ,iQ]=Mii[iQ]+tmp+mz
        matH[iQ+NK,iQ+NK]=Mii[iQ+NK]+tmp-mz

        tmp=(-kk[1]+Kvec[1,iQ])^2+(-kk[2]+Kvec[2,iQ])^2+u0
        matH[iQ+2*NK,iQ+2*NK]=-1*(Mii[iQ+2*NK]+tmp+mz)
        matH[iQ+3*NK,iQ+3*NK]=-1*(Mii[iQ+3*NK]+tmp-mz)
    end
    nothing
end

function eig_BdG(lat::Lattice,ϕ::Vector{ComplexF64},uu::Float64,k::Array{Float64,2},Nb::Int; mu::Bool=true)
    (;NK,v0,mz,Kvec) = lat
    matH = zeros(ComplexF64,4*NK,4*NK)
    intBdgM!(matH,ϕ,lat)
    Mii,u0 = matHu0!(matH,lat)
    println("check_u0: ",uu-u0)
    u0 = mu ? v0-u0 : uu    #  使用平均场基态求得的化学势或无能隙要求的化学势，默认后者

    en=Array{Float64,2}(undef,2*Nb,size(k,2))
    mtmp=similar(matH)
    for ik in axes(k,2)
        mtmp.=matH
        calmatH!(mtmp,Mii,k[:,ik],Kvec,NK,mz,u0)
        entmp=real.(eigvals!(mtmp))
        pt=partialsortperm(entmp,2*NK-Nb+1:2*NK+Nb)
        pt.=pt[[Nb+1:2*Nb; Nb:-1:1]]
        en[:,ik].=entmp[pt]
    end

    return en
end

function eig_BdG(lat::Lattice,ϕ::Vector{ComplexF64},uu::Float64,bz::Array{Float64,3},Nb::Int; mu::Bool=true)
    (;NK,v0,mz,Kvec) = lat
    matH = zeros(ComplexF64,4*NK,4*NK)
    intBdgM!(matH,ϕ,lat)
    Mii,u0 = matHu0!(matH,lat)
    println("check_u0: ",uu-u0)
    u0 = mu ? v0-u0 : uu    #  使用平均场基态求得的化学势或无能隙要求的化学势，默认后者

    en = Array{Float64}(undef,2*Nb,size(bz,2),size(bz,3))
    mtmp=similar(matH)
    for iy in axes(bz,3),ix in axes(bz,2)
        mtmp.=matH
        calmatH!(mtmp,Mii,bz[:,ix,iy],Kvec,NK,mz,u0)
        entmp=real.(eigvals!(mtmp))
        pt=partialsortperm(entmp,2*NK-Nb+1:2*NK+Nb)
        pt.=pt[[Nb+1:2*Nb; Nb:-1:1]]
        en[:,ix,iy].=entmp[pt]
    end

    return en
end

function eigen_BdG(lat::Lattice,ϕ::Vector{ComplexF64},uu::Float64,k::Array{Float64,2},Nb::Int; mu::Bool=true)
    (;NK,v0,mz,Kvec) = lat
    matH = zeros(ComplexF64,4*NK,4*NK)
    intBdgM!(matH,ϕ,lat)
    Mii,u0 = matHu0!(matH,lat)
    println("check_u0: ",uu-u0)
    u0 = mu ? v0-u0 : uu    #  使用平均场基态求得的化学势或无能隙要求的化学势，默认后者

    en=Array{Float64}(undef,2*Nb,size(k,2))
    ev=Array{ComplexF64}(undef,4*NK,2*Nb,size(k,2))
    mtmp=similar(matH)
    for ik in axes(k,2)
        mtmp.=matH
        calmatH!(mtmp,Mii,k[:,ik],Kvec,NK,mz,u0)
        _en,_ev = eigen!(mtmp)
        pt=partialsortperm(real.(_en),2*NK-Nb+1:2*NK+Nb)
        pt.=pt[[Nb+1:2*Nb; Nb:-1:1]]
        en[:,ik].=real.(_en[pt])
        _ev.=_ev[:,pt]
        norBev!(_ev)
        ev[:,:,ik].= _ev
    end

    return en,ev
end

function eigen_BdG(lat::Lattice,ϕ::Vector{ComplexF64},uu::Float64,bz::Array{Float64,3},Nb::Int; mu::Bool=true)
    (;NK,v0,mz,Kvec) = lat
    matH = zeros(ComplexF64,4*NK,4*NK)
    intBdgM!(matH,ϕ,lat)
    Mii,u0 = matHu0!(matH,lat)
    println("check_u0: ",uu-u0)
    u0 = mu ? v0-u0 : uu    #  使用平均场基态求得的化学势或无能隙要求的化学势，默认后者

    en=Array{Float64}(undef,2*Nb,size(bz,2),size(bz,3))
    ev=Array{ComplexF64}(undef,4*NK,2*Nb,size(bz,2),size(bz,3))
    mtmp=similar(matH)
    for iy in axes(bz,3),ix in axes(bz,2)
        mtmp.=matH
        calmatH!(mtmp,Mii,bz[:,ix,iy],Kvec,NK,mz,u0)
        _en,_ev = eigen!(mtmp)
        pt=partialsortperm(real.(_en),2*NK-Nb+1:2*NK+Nb)
        pt.=pt[[Nb+1:2*Nb; Nb:-1:1]]
        en[:,ix,iy].=real.(_en[pt])
        _ev.=_ev[:,pt]
        norBev!(_ev)
        ev[:,:,ix,iy].= _ev
    end

    return en,ev
end

# 计算k点的BdG矩阵，不左乘tz
function cal_BdG(lat::Lattice,ϕ::Vector{ComplexF64},uu::Float64,kk::Vector{Float64})
    matH=zeros(ComplexF64,4*lat.NK,4*lat.NK)
    intBdgM!(matH,ϕ,lat)
    Mii,u0=matHu0!(matH,lat)
    println("check_u0: ",uu-u0)
    u0=lat.v0-u0

    calmatH!(matH,Mii,kk,lat.Kvec,lat.NK,lat.mz,u0)
    tz=Diagonal([ones(2*lat.NK);fill(-1.0,2*lat.NK)])
    lmul!(tz,matH)

    return matH
end

function eigBdG(Mk0::Matrix{ComplexF64})
    Nm=round(Int,size(Mk0,1)/2)
    tz=Diagonal([ones(Nm);fill(-1.0,Nm)])

    ben,bev=eigen(tz*Mk0)
    pt=sortperm(real.(ben))
    pt.=[pt[Nm+1:end];reverse(pt[1:Nm])]
    ben.=ben[pt]
    bev.=bev[:,pt]

    norBev!(bev)
    gaugev!(bev)
    return ben,bev
end

function BdG_bcav(ev, bz)
    NK,Nb,Nx,Ny = size(ev)
    NK = rpund(Int, NK/2)
    ds=abs(det([bz[:,2,1].-bz[:,1,1] bz[:,1,2].-bz[:,1,1]]))
    tz=Diagonal([ones(NK); fill(-1.0,NK)])
    bcav = Array{Float64}(undef,Nb,ix,iy)
    Threads.@threads for iy in 1:Ny-1
        @views for ix in 1:Nx-1,ib in 1:Nb
            tmp=dot(ev[:,ib,ix,iy],tz,ev[:,ib,ix+1,iy])
            tmp*=dot(ev[:,ib,ix+1,iy],tz,ev[:,ib,ix+1,iy+1])
            tmp*=dot(ev[:,ib,ix+1,iy+1],tz,ev[:,ib,ix,iy+1])
            tmp*=dot(ev[:,ib,ix,iy+1],tz,ev[:,ib,ix,iy])
            bcav[ib,ix,iy]=-angle(tmp)/ds
        end
    end
    return bcav
end

