export Lattice,calmat,calVmat,eigband,eigenband,eigen2D,cal_Bcav,cal_bloch_wave,cal_bloch_spin
export dot_sx,dot_sy,dot_sz,dot_spin
export zeeman_split!

struct Lattice
    v0::Float64
    m0::Float64
    mz::Float64
    g1::Float64         # g_↑↑
    g2::Float64         # g_↑↓
    Kmax::Int           # 倒格子截断倍数

    b::Array{Float64,2} # 倒格子基矢
    a::Array{Float64,2} # 正格子基矢
    Sunit::Float64      # 元胞面积
    NK::Int             # 倒格矢数量
    Kcoe::Array{Int,2}        # 倒格矢展开系数 K[i] = Kcoe[j,i]b[:,j]
    Kvec::Array{Float64,2}    # 倒格矢集合 {K[i]}
end

"   Lattice(b, v0, m0, mz, g1, g2, Kmax) "
function Lattice(b,v0,m0,mz,g1,g2,Kmax)
    a = inv(b').*(2pi)
    Sunit=abs(det(a))
    NK,Kcoe,Kvec = CalKcoe(Kmax,b)
    Lattice(v0,m0,mz,g1,g2,Kmax,b,a,Sunit,NK,Kcoe,Kvec)
end

function CalKcoe(Kmax::Int, b::Array{Float64,2})
    Kcoe = Array{Int}(undef, 2, (2*Kmax)^2)
    Kvec = Array{Float64}(undef, 2, (2*Kmax)^2)
    @views pp = Kmax*max(norm(b[:,1]),norm(b[:,2]))+1e-6

    idx=1
    for ii in -Kmax:Kmax,jj in -Kmax:Kmax
        @views Kvec[:,idx].= ii.*b[:,1] .+ jj.*b[:,2]
        if norm(view(Kvec,:,idx))<pp
            Kcoe[:,idx].=ii,jj
            idx+=1
        end
    end
    idx-=1
    return idx,Kcoe[:,1:idx],Kvec[:,1:idx]
end

function _calKvec!(Kvec2,Kvec,kk)
    @inbounds for iQ in axes(Kvec,2)
        Kvec2[1,iQ]=Kvec[1,iQ]+kk[1]
        Kvec2[2,iQ]=Kvec[2,iQ]+kk[2]
    end
end

function calKvec(lat::Lattice,kk::AbstractArray{Float64})
    Kvec2=Array{Float64,2}(undef,2,lat.NK)
    _calKvec!(Kvec2,lat.Kvec,kk)
    return Kvec2
end

""" 
    cal_bloch_wave(kk::Vector, uk::Array{T,2}, lat, xx, yy)

计算k点多态 uk 的实空间布洛赫函数，注意自旋上和自旋下要分开计算
"""
function cal_bloch_wave(
    kk::Vector{Float64},
    uk::Array{<:Number,2},
    lat::Lattice,
    xx::AbstractVector{Float64},
    yy::AbstractVector{Float64}
)
    Nx = length(xx)
    Ny = length(yy)
    Nb = size(uk,2)
    NK = lat.NK

    Kvec = calKvec(lat,kk)
    plw=Array{ComplexF64,3}(undef,NK,Nx,Ny) #实空间中每个倒格矢的平面波
    Threads.@threads for iy in 1:Ny
        @inbounds for ix in 1:Nx,iQ in 1:NK
            plw[iQ,ix,iy]=cis(Kvec[1,iQ]*xx[ix]+Kvec[2,iQ]*yy[iy])
        end
    end

    wave=zeros(ComplexF64,Nx,Ny,Nb)
    for ib in 1:Nb
        Threads.@threads for iy in 1:Ny
            @inbounds for ix in 1:Nx,iQ in 1:NK
                wave[ix,iy,ib]+=uk[iQ,ib]*plw[iQ,ix,iy]
            end
        end
    end

    tmp=1/sqrt(lat.Sunit)
    Threads.@threads for ii in eachindex(wave)
        @inbounds wave[ii]*=tmp
    end
    return wave
end

""" 
    cal_bloch_wave(kk::Vector, uk::Array{T,1}, lat, xx, yy)

计算k点某个态 uk 的实空间布洛赫函数，注意自旋上和自旋下要分开计算
"""
function cal_bloch_wave(
    kk::Vector{Float64},
    uk::Array{<:Number,1},
    lat::Lattice,
    xx::AbstractVector{Float64},
    yy::AbstractVector{Float64}
)
    uk2 = reshape(uk,:,1)
    w = cal_bloch_wave(kk,uk2,lat,xx,yy)
    dropdims(w, dims=3)
end

""" 
    cal_bloch_wave(kk::Vector, uk::Array{T,2}, lat, bz::Array{T,3})

计算k点多个态 uk 的实空间bz区域上的布洛赫函数，自旋上和自旋下要分开计算
"""
function cal_bloch_wave(
    kk::Vector{Float64},
    uk::Array{<:Number,2},
    lat::Lattice,
    bz::Array{Float64,3}
)
    _,Nx,Ny=size(bz)
    Nb=size(uk,2)
    NK=lat.NK

    Kvec=calKvec(lat,kk)

    plw=Array{ComplexF64,3}(undef,NK,Nx,Ny) #实空间中每个倒格矢的平面波
    Threads.@threads for iy in 1:Ny
        @inbounds for ix in 1:Nx,iQ in 1:NK
            plw[iQ,ix,iy]=cis(Kvec[1,iQ]*bz[1,ix,iy]+Kvec[2,iQ]*bz[2,ix,iy])
        end
    end
    wave=zeros(ComplexF64,Nx,Ny,Nb)
    for ib in 1:Nb
        Threads.@threads for iy in 1:Ny
            @inbounds for ix in 1:Nx,iQ in 1:NK
                wave[ix,iy,ib]+=uk[iQ,ib]*plw[iQ,ix,iy]
            end
        end
    end

    tmp=1/sqrt(lat.Sunit)
    Threads.@threads for ii in eachindex(wave)
        @inbounds wave[ii]*=tmp
    end

    return wave
end

""" 
    cal_bloch_wave(kk::Vector, uk::Array{T,1}, lat, bz::Array{T,3})

计算k点单个态 uk 的实空间bz区域上的布洛赫函数，自旋上和自旋下要分开计算
"""
function cal_bloch_wave(
    kk::Vector{Float64},
    uk::Array{<:Number,1},
    lat::Lattice,
    bz::Array{Float64,3}
)
    uk2 = reshape(uk,:,1)
    w = cal_bloch_wave(kk,uk2,lat,bz)
    dropdims(w, dims=3)
end

function cal_bloch_spin(
    kk::Vector{Float64},
    uk::Array{<:Number,1},
    lat::Lattice,
    xx::AbstractVector{Float64},
    yy::AbstractVector{Float64}
)
    wup = cal_bloch_wave(kk, uk[1:lat.NK], lat, xx, yy)
    wdn = cal_bloch_wave(kk, uk[lat.NK+1:end], lat, xx, yy)
    sx = Array{Float64}(undef,length(xx),length(yy))
    sy = similar(sx)
    sz = similar(sx)
    Threads.@threads for ii in eachindex(wup)
        sx[ii] = 2*real(conj(wup[ii])*wdn[ii])
        sy[ii] = 2*imag(conj(wup[ii])*wdn[ii])
        sz[ii] = abs2(wup[ii])-abs2(wdn[ii])
    end
    return sx,sy,sz
end

# 哈密顿矩阵非对角元
# b1 = [1,1], b2=[-1,1]
# V_{11} = V[cos^2(x)+cos^2(y)]
#        = V/4(e^{i2x}+e^{-i2x}+e^{i2y}+e^{-i2y})+V
#        = V/4(e^{i(b1-b2)}+e^{i(-b1+b2)}+e^{i(b1+b2)}+e^{i(-b1-b2)})+V
# V_{12} = M[sin(x)cos(y)-isin(y)cos(x)]
#        = M/(4i)[e^{i(x+y)}+e^{i(x-y)}-e^{i(-x+y)}-e^{i(-x-y)}]-M/4(x↔y)
#        = M/4[-(i+1)e^{ib1}+(-i+1)e^{-ib2}+(i-1)e^{ib2}+(i+1)e^{-ib1}]
function matoff!(mat::Matrix{ComplexF64},v0::Float64,m0::Float64,Kcoe::Array{Int,2})
    NK=size(Kcoe,2)

    V11 = ((1,-1), (-1,1), (1,1), (-1,-1))
    V12 = ((1,0), (-1,0), (0,1), (0,-1))
    M = (-1im-1.0, 1im+1.0, 1im-1.0, -1im+1.0).*(m0/4)

    @inbounds for jj in 1:NK,ii in 1:NK
        t1 = Kcoe[1,ii]-Kcoe[1,jj]
        t2 = Kcoe[2,ii]-Kcoe[2,jj]

        for iv in V11
            if iv[1]==t1 && iv[2]==t2
                mat[ii,jj]=mat[ii+NK,jj+NK] = v0/4.0
                break
            end
        end

        for (i,iv) in enumerate(V12)
            if iv[1]==t1 && iv[2]==t2
                mat[ii,jj+NK] = M[i]
                break
            end
        end
    end
    mat .= Hermitian(mat)
    return mat
end

# 无自旋轨道耦合哈密顿量, 布里渊区扩大一倍
function matoff2!(mat::Matrix{ComplexF64},v0::Float64,m0::Float64,Kcoe::Array{Int,2})
    NK=size(Kcoe,2)
    V11 = ((1,0), (-1,0), (0,1), (0,-1))

    @inbounds for jj in 1:NK,ii in 1:NK
        t1 = Kcoe[1,ii]-Kcoe[1,jj]
        t2 = Kcoe[2,ii]-Kcoe[2,jj]
        for iv in V11
            if iv[1]==t1 && iv[2]==t2
                mat[ii,jj]=mat[ii+NK,jj+NK] = v0/4.0
                break
            end
        end
    end
    return mat
end

# 哈密顿矩阵对角元
function matdiag!(
    mat::Matrix{ComplexF64},
    kk::AbstractVector{Float64},
    Kvec::Array{Float64,2},
    v0::Float64,
    mz::Float64
)
    NK = size(Kvec,2)
    for ii in 1:NK
        tmp = (kk[1]+Kvec[1,ii])^2+(kk[2]+Kvec[2,ii])^2+v0
        mat[ii,ii]=tmp+mz
        mat[ii+NK,ii+NK]=tmp-mz
    end
    nothing
end

"""
    calDhk(lat, k; u::Int=1)
对k点点h(k)关于u方向求导
"""
function calDhk(lat::Lattice,k::AbstractVector{Float64}; u::Int=1)
    (;NK,Kvec) = lat
    Dhk = Array{ComplexF64}(undef,2*NK)
    for ii in 1:NK
        Dhk[ii] = (k[u]+Kvec[u,ii])*2
        Dhk[ii+NK] = Dhk[ii]
    end
    return Diagonal(Dhk)
end


"""
    calmat(lat, kk)
计算kk点的单粒子哈密顿矩阵
"""
function calmat(lat::Lattice,kk::Array{Float64,1})
    mat = zeros(ComplexF64, 2*lat.NK,2*lat.NK)
    matoff!(mat, lat.v0, lat.m0, lat.Kcoe)
    matdiag!(mat, kk, lat.Kvec, lat.v0, lat.mz)
    return mat
end

"""
    calVmat(lat, kk)
计算哈密顿矩阵的势能部分
"""
function calVmat(lat::Lattice)
    mat = zeros(ComplexF64, 2*lat.NK,2*lat.NK)
    matoff!(mat, lat.v0, lat.m0, lat.Kcoe)
    Vup = lat.v0+lat.mz
    Vdn = lat.v0-lat.mz
    for ii in 1:lat.NK
        mat[ii,ii] = Vup
        mat[ii+lat.NK,ii+lat.NK] = Vdn
    end
    return mat
end

"""
    eigband(lat, kk::Matrix, nb)
计算多个动量点的本征能量
"""
function eigband(lat::Lattice,kk::Array{Float64,2},nb::AbstractVector{Int})
    mat=zeros(ComplexF64,2*lat.NK,2*lat.NK)
    matoff!(mat,lat.v0,lat.m0,lat.Kcoe)
    Nk=size(kk,2)
    en=Array{Float64,2}(undef,nb[end]-nb[1]+1,Nk)
    
    @inbounds for ik in 1:Nk
        matdiag!(mat,kk[:,ik],lat.Kvec,lat.v0,lat.mz)
        en[:,ik].= eigvals(Hermitian(mat),nb[1]:nb[end])
    end
    return en
end

"""
    eigband(lat, bz::Matrix, nb)
计算多个动量点的本征能量
"""
function eigband(lat::Lattice,bz::Array{Float64,3},nb::AbstractVector{Int})
    mat=zeros(ComplexF64,2*lat.NK,2*lat.NK)
    matoff!(mat,lat.v0,lat.m0,lat.Kcoe)
    _,Nx,Ny = size(bz)
    en=Array{Float64}(undef,nb[end]-nb[1]+1,Nx,Ny)
    
    @inbounds for iy in 1:Ny,ix in 1:Nx
        matdiag!(mat,bz[:,ix,iy],lat.Kvec,lat.v0,lat.mz)
        en[:,ix,iy].= eigvals(Hermitian(mat),nb[1]:nb[end])
    end
    return en
end

"""
    eigband(lat, kk::Vector, nb)
计算单个动量点的本征能量
"""
function eigband(lat::Lattice,kk::Vector{Float64},nb::AbstractVector{Int})
    k2 = reshape(kk,:,1)
    reshape(eigband(lat,k2,nb),:)
end

"""
    eigband(lat, kk::Matrix)
计算多个动量点的本征能量和态
"""
function eigenband(lat::Lattice,kk::Array{Float64,2},nb::AbstractVector{Int})
    mat=zeros(ComplexF64,2*lat.NK,2*lat.NK)
    matoff!(mat,lat.v0,lat.m0,lat.Kcoe)

    Nk=size(kk,2)
    Nb=nb[end]-nb[1]+1
    en=Array{Float64,2}(undef,Nb,Nk)
    ev=Array{ComplexF64,3}(undef,2*lat.NK,Nb,Nk)

    mz = abs(lat.mz)<1e-9
    @inbounds for ik in 1:Nk
        @views matdiag!(mat,kk[:,ik],lat.Kvec, lat.v0, lat.mz)
        en_tmp,ev_tmp=eigen(Hermitian(mat))
        pt = partialsortperm(en_tmp,nb[1]:nb[end])
        en[:,ik].= en_tmp[pt]
        ev[:,:,ik].= ev_tmp[:,pt]
        mz && zeeman_split!(view(ev,:,:,ik))
    end
    return en,ev
end

"""
    eigband(lat, kk::Matrix)
计算多个动量点的本征能量和态
"""
function eigenband(lat::Lattice,kk::Array{Float64,3},nb::AbstractVector{Int})
    mat=zeros(ComplexF64,2*lat.NK,2*lat.NK)
    matoff!(mat,lat.v0,lat.m0,lat.Kcoe)

    _,Nx,Ny=size(kk)
    Nb=nb[end]-nb[1]+1
    en=Array{Float64}(undef,Nb,Nx,Ny)
    ev=Array{ComplexF64}(undef,2*lat.NK,Nb,Nx,Ny)

    mz = abs(lat.mz)<1e-9
    @inbounds for iy in 1:Ny, ix in 1:Nx
        @views matdiag!(mat,kk[:,ix,iy],lat.Kvec, lat.v0, lat.mz)
        en_tmp,ev_tmp=eigen(Hermitian(mat))
        pt = partialsortperm(en_tmp,nb[1]:nb[end])
        en[:,ix,iy].= en_tmp[pt]
        ev[:,:,ix,iy].= ev_tmp[:,pt]
        mz && zeeman_split!(view(ev,:,:,ix,iy))
    end
    return en,ev
end

"""
    eigband(lat, kk::Vector)
计算单个动量点的本征能量和态
"""
function eigenband(lat::Lattice,kk::Array{Float64,1},nb::AbstractVector{Int})
    k2 = reshape(kk,:,1)
    en,ev=eigenband(lat,k2,nb)
    ev2 = dropdims(ev,dims=3)
    gaugev!(ev2)
    dropdims(en,dims=2),ev2
end

"""
    eigen2D(lat ,bz, nb)
计算区域bz的本征能量，态和贝里曲率
"""
function eigen2D(lat::Lattice,bz::Array{Float64,3},nb::AbstractVector{Int})
    mat=zeros(ComplexF64,2*lat.NK,2*lat.NK)
    matoff!(mat,lat.v0,lat.m0,lat.Kcoe)
    _,Nx,Ny=size(bz)
    Nb = nb[end]-nb[1]+1

    en=Array{Float64,3}(undef,Nb,Nx,Ny)
    ev=Array{ComplexF64,4}(undef,2*lat.NK,Nb,Nx,Ny)

    mz = abs(lat.mz)<1e-10
    for iy in 1:Ny,ix in 1:Nx
        @views matdiag!(mat,bz[:,ix,iy],lat.Kvec,lat.v0,lat.mz)
        en_tmp,ev_tmp=eigen(Hermitian(mat))
        pt = partialsortperm(en_tmp,nb[1]:nb[end])
        ev_tmp2=ev_tmp[:,pt]
        mz && zeeman_split!(ev_tmp2)
        en[:,ix,iy].= en_tmp[pt]
        ev[:,:,ix,iy].= ev_tmp2
    end

    bcav=Array{Float64,3}(undef,Nb,Nx-1,Ny-1)
    ds=abs(det([bz[:,2,1].-bz[:,1,1] bz[:,1,2].-bz[:,1,1]]))
    @views for iy in 1:Ny-1,ix in 1:Nx-1,ib in 1:Nb
        tmp=dot(ev[:,ib,ix,iy],ev[:,ib,ix+1,iy])
        tmp*=dot(ev[:,ib,ix+1,iy],ev[:,ib,ix+1,iy+1])
        tmp*=dot(ev[:,ib,ix+1,iy+1],ev[:,ib,ix,iy+1])
        tmp*=dot(ev[:,ib,ix,iy+1],ev[:,ib,ix,iy])
        bcav[ib,ix,iy]=-angle(tmp)/ds
    end
    return (;en,ev,bcav)
end

"""
    cal_Bcav(lat,kk,nb)
计算某个点kk的贝里曲率, hk求导法
"""
function cal_Bcav(lat::Lattice,kk::AbstractVector{Float64},nb::AbstractVector{Int})
    NK=lat.NK
    mat=calmat(lat,kk)

    en,ev=eigen(Hermitian(mat))
    zeeman_split!(ev)

    Dhx = calDhk(lat,kk; u=1)
    Dhy = calDhk(lat,kk; u=2)

    bcav=Array{Float64}(undef,nb[end]-nb[1]+1)
    for ib in nb[1]:nb[end]
        tmp=0.0
        @views for ii in 1:2*NK
            dE=en[ii]-en[ib]
            abs(dE)<1e-6 && continue
            tmp+=imag(dot(ev[:,ib],Dhx,ev[:,ii])*dot(ev[:,ii],Dhy,ev[:,ib]))/dE^2
        end
        bcav[ib]=-2*tmp
    end

    return bcav
end

 
"""
    dot_sz(ψ, ϕ)
计算矩阵元⟨ψ|σ_z|ϕ⟩
"""
function dot_sz(ψ::AbstractVector{ComplexF64},ϕ::AbstractVector{ComplexF64})
    lenev = div(length(ψ),2)
    sgz = 0.0im
    for ii in 1:lenev
        sgz+=conj(ψ[ii])*ϕ[ii]
    end
    for ii in lenev+1:2*lenev
        sgz-=conj(ψ[ii])*ϕ[ii]
    end
    return sgz
end

"""
    dot_sz(ϕ)
计算期望值⟨ϕ|σ_z|ϕ⟩
"""
dot_sz(ϕ::AbstractVector{ComplexF64})=dot_sz(ϕ,ϕ)

"""
    dot_sx(ψ, ϕ)
计算矩阵元⟨ψ|σ_x|ϕ⟩
"""
function dot_sx(ψ::AbstractVector{ComplexF64},ϕ::AbstractVector{ComplexF64})
    lenev=div(length(ψ),2)
    sgx = 0.0im
    for ii in 1:lenev
        sgx+=conj(ψ[ii])*ϕ[ii+lenev]
        sgx+=conj(ψ[ii+lenev])*ϕ[ii]
    end
    return sgx
end

"""
    dot_sx(ϕ)
计算期望值⟨ϕ|σ_x|ϕ⟩
"""
dot_sx(ev::AbstractVector{ComplexF64})=dot_sx(ev,ev)

"dot_sy(ψ, ϕ), 计算矩阵元⟨ψ|σ_y|ϕ⟩"
function dot_sy(ψ::AbstractVector{ComplexF64},ϕ::AbstractVector{ComplexF64})
    lenev=div(length(ψ),2)
    sgy = 0.0im
    for ii in 1:lenev
        sgy+=(conj(ψ[ii+lenev])*ϕ[ii]-conj(ψ[ii])*ϕ[ii+lenev])*1im
    end
    return sgy
end

"""
    dot_sy(ϕ)
计算期望值⟨ϕ|σ_y|ϕ⟩
"""
dot_sy(ev::AbstractVector{ComplexF64})=dot_sy(ev,ev)

"""
    dot_sxy(ψ, ϕ)
计算矩阵元⟨ψ|σ_x+σ_y|ϕ⟩/√2
"""
function dot_sxy(ψ::AbstractVector{ComplexF64},ϕ::AbstractVector{ComplexF64})
    lenev=div(length(ψ),2)
    s1=s2=0.0im
    for ii in 1:lenev
        s1+=conj(ψ[ii+lenev])*ϕ[ii]
        s2+=conj(ψ[ii])*ϕ[ii+lenev]
    end
    s1=((1+1.0im)*s1+(1-1.0im)*s2)/√2
    return s1
end

"""
    dot_sxy(ϕ)
计算期望值⟨ϕ|σ_xy|ϕ⟩
"""
dot_sxy(ev::AbstractVector{ComplexF64})=dot_sxy(ev,ev)

"""
    dot_spin(ψ, ϕ)
计算矩阵元⟨ψ|σ_i|ϕ⟩, i=x,y,z
"""
function dot_spin(ψ::AbstractVector{ComplexF64},ϕ::AbstractVector{ComplexF64})
    sp = Vector{ComplexF64}(undef,3)
    sp[1] = dot_sx(ψ,ϕ)
    sp[2] = dot_sy(ψ,ϕ)
    sp[3] = dot_sz(ψ,ϕ)
    return sp
end

"""
    dot_spin(ϕ, ϕ)
计算 期望值⟨ϕ|σ_i|ϕ⟩, i=x,y,z
"""
dot_spin(ψ::AbstractVector{ComplexF64})=dot_spin(ψ,ψ)

"""
    dot_spin(ψ↑, ψ↓)
计算自旋平均值
"""
function dot_spin(ψ::ComplexF64,ϕ::ComplexF64)
    ss=abs2(ψ)+abs2(ϕ)
    sgx=2*real(conj(ψ)*ϕ)/ss
    sgy=2*imag(conj(ψ)*ϕ)/ss
    sgz=(abs2(ψ)-abs2(ϕ))/ss
    return sgx,sgy,sgz
end

"""
    zeeman_split!(ev::AbstractArray{ComplexF64,2})
塞曼分裂计算分离简并
"""
function zeeman_split!(ev::AbstractArray{ComplexF64,2})
    Nb=size(ev,2)
    @assert iseven(Nb) "sigmazeig error"
    Nb=div(Nb,2)
    pertu=Array{ComplexF64}(undef, 2, 2)

    vtmp=Array{ComplexF64}(undef,size(ev,1),2)
    @views for ii in 1:Nb
        pertu[1,1]=dot_sz(ev[:,2*ii-1],ev[:,2*ii-1])
        pertu[2,1]=dot_sz(ev[:,2*ii],ev[:,2*ii-1])
        pertu[1,2]=dot_sz(ev[:,2*ii-1],ev[:,2*ii])
        pertu[2,2]=dot_sz(ev[:,2*ii],ev[:,2*ii])
        
        en_tmp,ev_tmp=eigen(Hermitian(pertu))
        vtmp[:,1] .= ev_tmp[1,1].*ev[:,2*ii-1].+ev_tmp[2,1].*ev[:,2*ii]
        vtmp[:,2] .= ev_tmp[1,2].*ev[:,2*ii-1].+ev_tmp[2,2].*ev[:,2*ii]
    
        if en_tmp[1]<en_tmp[2]
            ev[:,2*ii-1].=vtmp[:,1] 
             ev[:,2*ii].=vtmp[:,2]
        else
            ev[:,2*ii-1].=vtmp[:,2]
            ev[:,2*ii].=vtmp[:,1] 
        end 
    end
end

function zeeman_split_sxy!(ev::AbstractArray{ComplexF64,2})
    Nb=size(ev,2)
    @assert iseven(Nb) "sigmazeig error"
    Nb=round(Int,Nb/2)
    pertu=Array{ComplexF64,2}(undef,2,2)

    vtmp=Array{ComplexF64}(undef,size(ev,1),2)
    @views for ii in 1:Nb
        pertu[1,1]=dot_sxy(ev[:,2*ii-1],ev[:,2*ii-1])
        pertu[2,1]=dot_sxy(ev[:,2*ii],ev[:,2*ii-1])
        pertu[1,2]=dot_sxy(ev[:,2*ii-1],ev[:,2*ii])
        pertu[2,2]=dot_sxy(ev[:,2*ii],ev[:,2*ii])
        
        en_tmp,ev_tmp=eigen(Hermitian(pertu))
        vtmp[:,1].=ev_tmp[1,1].*ev[:,2*ii-1].+ev_tmp[2,1].*ev[:,2*ii]
        vtmp[:,2].=ev_tmp[1,2].*ev[:,2*ii-1].+ev_tmp[2,2].*ev[:,2*ii]
    
        if en_tmp[1]<en_tmp[2]
            ev[:,2*ii-1].=vtmp[:,1]
            ev[:,2*ii].=vtmp[:,2]
        else
            ev[:,2*ii-1].=vtmp[:,2]
            ev[:,2*ii].=vtmp[:,1]
        end 
    end
    # gaugev!(ev)
end

