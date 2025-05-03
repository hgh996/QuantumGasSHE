#---------------------------------------------------------
#           Hall conductivity
#---------------------------------------------------------
export cal_Ju,cal_Jθ,Green1,eigBdG,Xspec1,Xspec2,Hall2ed,FermiHall,Hall_time,cal_Dθ,GreenX

function cal_Ju(
    ϕ::Vector{ComplexF64},
    kk::Vector{Float64},
    Kvec::Array{Float64,2};
    u::Int=1, # u=1,2 分别为x，y方向
    sp::Int=1 # sp=±1 代表粒子流和自旋流
)
    NQ=size(Kvec,2)
    Nm=NQ*2
    Ju=Vector{ComplexF64}(undef,Nm*2)
    Dhu=similar(Ju)

    for iQ in 1:NQ
        Dhu[iQ]=2*(kk[u]+Kvec[u,iQ])
        Dhu[iQ+NQ]=Dhu[iQ]*sp

        Dhu[iQ+Nm]=2*(-kk[u]+Kvec[u,iQ])
        Dhu[iQ+Nm+NQ]=Dhu[iQ+Nm]*sp
    end
    for iQ in 1:Nm
        Ju[iQ]=Dhu[iQ]*ϕ[iQ]
        Ju[iQ+Nm]=Dhu[iQ+Nm]*conj(ϕ[iQ])
    end

    return Ju,Diagonal(Dhu)
end

# diag[∇h(k), ∇h^*(-k)]
function cal_Dθ(
    kk::AbstractVector{Float64},
    Kvec::Array{Float64,2},
    θ::Float64;
    d::Float64=1e-4, #  中心差分步长
    sp::Int=1        # sp=±1 代表粒子流和自旋流
)
    NQ=size(Kvec,2)
    Nm=NQ*2
    Dhu=Vector{Float64}(undef,Nm*2)

    k1 = (kk[1]+d*cos(θ),kk[2]+d*sin(θ))
    k2 = (kk[1]-d*cos(θ),kk[2]-d*sin(θ))

    for iQ in 1:NQ
        tmp = (k1[1]+Kvec[1,iQ])^2+(k1[2]+Kvec[2,iQ])^2
        tmp-= (k2[1]+Kvec[1,iQ])^2+(k2[2]+Kvec[2,iQ])^2
        Dhu[iQ]= tmp/(2*d)
        Dhu[iQ+NQ]=Dhu[iQ]*sp

        tmp = (-k1[1]+Kvec[1,iQ])^2+(-k1[2]+Kvec[2,iQ])^2
        tmp-= (-k2[1]+Kvec[1,iQ])^2+(-k2[2]+Kvec[2,iQ])^2
        Dhu[iQ+Nm]=tmp/(-2*d)
        Dhu[iQ+Nm+NQ]=Dhu[iQ+Nm]*sp
    end
    return Diagonal(Dhu)
end

function cal_Jθ(
    ϕ::Vector{ComplexF64},
    kk::Vector{Float64},
    Kvec::Array{Float64,2},
    θ::Float64;
    d::Float64=1e-4, #  中心差分步长
    sp::Int=1        # sp=±1 代表粒子流和自旋流
)
    Nm = size(Kvec,2)*2 
    Dhu = cal_Dθ(kk,Kvec,θ; sp=sp,d=d)
    Ju = Vector{ComplexF64}(undef,Nm*2)
    for iQ in 1:Nm
        Ju[iQ] = Dhu[iQ,iQ]*ϕ[iQ]
        Ju[iQ+Nm] = Dhu[iQ+Nm,iQ+Nm]*conj(ϕ[iQ])
    end
    return Ju
end

#  finite frequency Green function
function Green1(
    Mk0::Matrix{ComplexF64},
    w::AbstractVector{Float64},
    Jx::AbstractVector{ComplexF64},
    Jy::AbstractVector{ComplexF64};
    η::Float64=0.0
)
    Nm=round(Int,size(Mk0,1)/2)
    Nw=length(w)
    Xw=Vector{ComplexF64}(undef,Nw)

    Gw=-1.0.*Mk0
    for iw in 1:Nw
        ww = abs(w[iw])>1e-4 ? w[iw]+1im*η : 1e-4+1im*η
        for iQ in 1:Nm
            Gw[iQ,iQ]=ww-Mk0[iQ,iQ]
            Gw[iQ+Nm,iQ+Nm]=-ww-Mk0[iQ+Nm,iQ+Nm]
        end
        Xw[iw]=dot(Jx,inv(Gw),Jy)*1im/ww
    end
    return Xw
end

# 不除 ω
function GreenX(
    Mk0::Matrix{ComplexF64},
    w::AbstractVector{Float64},
    Jx::AbstractVector{ComplexF64},
    Jy::AbstractVector{ComplexF64};
    η::Float64=0.0
)
    Nm=round(Int,size(Mk0,1)/2)
    Nw=length(w)
    Xw=Vector{ComplexF64}(undef,Nw)

    Gw=-1.0.*Mk0
    for iw in 1:Nw
        # ww = abs(w[iw])>1e-4 ? w[iw]+1im*η : 1e-4+1im*η
        ww = w[iw]+1im*η
        for iQ in 1:Nm
            Gw[iQ,iQ]=ww-Mk0[iQ,iQ]
            Gw[iQ+Nm,iQ+Nm]=-ww-Mk0[iQ+Nm,iQ+Nm]
        end
        Xw[iw]=dot(Jx,inv(Gw),Jy)
    end
    return Xw
end

#  zero-frequency Green function
function Green1(
    Mk0::Matrix{ComplexF64},
    Jx::AbstractVector{ComplexF64},
    Jy::AbstractVector{ComplexF64};
    η::Float64=0.0,
    w::Float64=1e-4
)
    Nm=round(Int,size(Mk0,1)/2)
    w2 = [w,-w]
    Xw = Array{ComplexF64}(undef,2)
    Gw=-1.0.*Mk0
    for iw in 1:2
        ww=w2[iw]+1im*η
        for iQ in 1:Nm
            Gw[iQ,iQ]=ww-Mk0[iQ,iQ]
            Gw[iQ+Nm,iQ+Nm]=-ww-Mk0[iQ+Nm,iQ+Nm]
        end
        Xw[iw]=dot(Jx,inv(Gw),Jy)
    end

    return (Xw[1]-Xw[2])/(2*η-2im*w)
end


# 谱分解计算响应
function Xspec1(w::AbstractArray{Float64},Hx,Hy,ben,bev,ϕG; η::Float64=0.0)
    Nm=round(Int,length(ben)/2)
    Nw=length(w)
    Xw=zeros(ComplexF64,Nw)

    v0=[ϕG; conj.(ϕG)]
    E0=ben[1]
    for iw in 1:Nw
        ww = abs(w[iw])>5e-6 ? w[iw]+1im*η : 5e-6+1im*η
        tmp=0.0im
        @views for nn in 2:Nm
            tmp+=dot(v0,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0)/(ww-ben[nn]+E0)
            tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v0)/(ww+ben[nn]-E0)
        end
        Xw[iw]=tmp*1im/ww
    end
    return Xw
end

# ⟨ϕ|σ|v₂⟩×⟨v₂|j|vₙ⟩⟨vₙ|j|ϕ⟩，有限频
function Xspec2(w::AbstractArray{Float64},Hx,Hy,ben,bev,ϕG; η::Float64=0.0)
    Nm=round(Int,length(ben)/2)
    Nw=length(w)
    Xw=zeros(ComplexF64,Nw)

    v0=[ϕG; conj.(ϕG)]
    v1=similar(v0)
    if abs(ben[2]-ben[1])<1e-6
        v1.=[ϕG2; zeros(ComplexF64, Nm)]
    else
        v1.=bev[:,2]
    end
    E0=ben[1]

    NK = round(Int,length(ben)/4)
    sz = Diagonal([ones(NK);fill(-1.0,NK);ones(NK);fill(-1.0,NK)])
    mz = dot(v0,sz,v1)
    for iw in 1:Nw
        ww = abs(w[iw])>5e-5 ? w[iw]+1im*η : 5e-5+1im*η
        tmp=0.0im
        @views for nn in 2:Nm
            tmp+=mz*dot(v1,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0)/(ww-ben[nn]+E0)
            tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v1)*conj(mz)/(ww+ben[nn]-E0)
        end
        Xw[iw]=tmp*1im/ww
    end
    return Xw
end

# ⟨ϕ|σ|v₂⟩×⟨v₂|j|vₙ⟩⟨vₙ|j|ϕ⟩,零频
function Xspec2(Hx,Hy,ben,bev,ϕG,ϕG2)
    Nm=round(Int,length(ben)/2)
    v0=[ϕG; conj.(ϕG)]
    v1=similar(v0)
    if abs(ben[2]-ben[1])<1e-6
        v1.=[ϕG2; zeros(ComplexF64, Nm)]
    else
        v1.=bev[:,2]
    end
    E0=ben[1]

    NK = round(Int,length(ben)/4)
    sz = Diagonal([ones(NK);fill(-1.0,NK);ones(NK);fill(-1.0,NK)])
    mz = dot(v0,sz,v1)
    println(mz)

    tmp=0.0im
    @views for nn in 2:Nm
        w = nn==2 ? 1e-6 : 0.0
        tmp+=2*imag(mz*dot(v1,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0))/(w+E0-ben[nn])^2
        # tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v1)*conj(mz)/(w+ben[nn]-E0)^2
    end

    return tmp
end

# ---------------------------------------------
#     second order Bogoliubov contribution
# ---------------------------------------------
function _Hall2ed(en,ev,j1,j2,nE)
    Nb = round(Int, size(ev,1)/2)
    s = 0.0im
    v1 = Array{ComplexF64}(undef,2*Nb)
    v2 = similar(v1)
    @inbounds @views for n in 1:2*Nb
        v1 .= j1*ev[:,n]
        v2 .= j2*ev[:,n]
        for m in 1:2*Nb
            a = nE[m]-nE[n]
            a == 0 && continue
            s-=a*dot(v1,ev[:,m])*dot(ev[:,m],v2)/(en[m]-en[n])^2
        end
    end
    return 0.5im*s
end

function Hall2ed(en,ev,bz,Kvec;θ::Tuple{Float64,Float64}=(0.0,pi/2),sp::Tuple{Int,Int}=(-1,1))
    Nb = round(Int, size(en,1)/2)
    nE = vcat(zeros(Int,Nb),fill(-1,Nb))
    s = Array{ComplexF64}(undef,size(en,2),size(en,3))
    Threads.@threads for iy in axes(en,3)
        @views for ix in axes(en,2)
            Dhx = cal_Dθ(bz[:,ix,iy],Kvec,θ[1]; sp=sp[1])
            Dhy = cal_Dθ(bz[:,ix,iy],Kvec,θ[2]; sp=sp[2])
            s[ix,iy] = _Hall2ed(en[:,ix,iy],ev[:,:,ix,iy],Dhx,Dhy,nE)
        end
    end
    return s
end

function Qudep(ev)
    _,Nb,Nx,Ny = size(ev)
    Nb = round(Int, Nb/2)
    ndep = Array{ComplexF64}(undef,Nx,Ny)
    Threads.@threads for iy in axes(ev,4)
        for ix in axes(ev,3)
            s = -Nb+0.0im
            @views for ib in Nb+1:2*Nb
                s+=dot(ev[:,ib,ix,iy],ev[:,ib,ix,iy])
            end
            ndep[ix,iy] = s/2
        end
    end
    return ndep
end

# ---------------------------------------
#       nointeracting Fermi gas
# ---------------------------------------
function cal_Jθ(
    kk::Vector{Float64},
    Kvec::Array{Float64,2},
    θ::Float64;
    d::Float64=1e-4, #  中心差分步长
    sp::Int=1        # sp=±1 代表粒子流和自旋流
)
    Ju=Vector{Float64}(undef,2*size(Kvec,2))
    k1 = (kk[1]+d*cos(θ),kk[2]+d*sin(θ))
    k2 = (kk[1]-d*cos(θ),kk[2]-d*sin(θ))
    NK = size(Kvec,2)
    for iQ in 1:NK
        tmp = (k1[1]+Kvec[1,iQ])^2+(k1[2]+Kvec[2,iQ])^2
        tmp-= (k2[1]+Kvec[1,iQ])^2+(k2[2]+Kvec[2,iQ])^2
        Ju[iQ]= tmp/(2*d)
        Ju[iQ+NK]=Ju[iQ]*sp
    end

    return Diagonal(Ju)
end

function _fermiHall(
    en::AbstractVector{Float64},
    ev::AbstractArray{<:Number,2},
    hx::AbstractMatrix{<:Number},
    hy::AbstractMatrix{<:Number},
    Nb::Int = 4,
    mu::Float64 = en[Nb]
)
    s_arr = zeros(Nb)
    nth = Threads.nthreads()
    sm_arr = Array{Float64}(undef,nth)
    chs = chunks(1:length(en), n=nth)

    @views for n in eachindex(s_arr)
        en[n]>mu-1e-12 && continue
        v1 = hx*ev[:,n]
        v2 = hy*ev[:,n]
        Threads.@threads for i in 1:nth
            s1 = 0.0
            @views for m in chs[i]
                en[m]<mu && continue
                dE = en[m]-en[n]
                abs(dE)<1e-5 && (dE = 1e-5)
                s1+= 2*imag(dot(v1,ev[:,m])*dot(ev[:,m],v2))/dE^2
            end
            sm_arr[i] = s1
        end
        s_arr[n]=sum(sm_arr)
    end
    return s_arr
end
function _fermiBerry(
    en::AbstractVector{Float64},
    ev::AbstractArray{<:Number,2},
    hx::AbstractMatrix{<:Number},
    hy::AbstractMatrix{<:Number},
    Nb::Int = 4,
    mu::Float64 = en[Nb]
)
    s_arr = zeros(Nb)
    nth = Threads.nthreads()
    sm_arr = Array{Float64}(undef,nth)
    chs = chunks(1:length(en), n=nth)

    @views for n in eachindex(s_arr)
        en[n]>mu-1e-12 && continue
        v1 = hx*ev[:,n]
        v2 = hy*ev[:,n]
        Threads.@threads for i in 1:nth
            s1 = 0.0
            @views for m in chs[i]
                m==n && continue
                dE = en[m]-en[n]
                abs(dE)<1e-5 && (dE = 1e-5)
                s1+= 2*imag(dot(v1,ev[:,m])*dot(ev[:,m],v2))/dE^2
            end
            sm_arr[i] = s1
        end
        s_arr[n]=sum(sm_arr)
    end
    return s_arr
end

function _fermiChi(
    w ::AbstractVector{Float64},
    en::AbstractVector{Float64},
    ev::AbstractArray{<:Number,2},
    hx::AbstractMatrix{<:Number},
    hy::AbstractMatrix{<:Number},
    Nb::Int = 4,
    mu::Float64 = en[Nb]+1e-11;
    η::ComplexF64 = 0.0im
)
    Xw = zeros(ComplexF64,length(w))
    @views for n in 1:Nb
        en[n]>mu-1e-12 && continue
        v1 = hx*ev[:,n]
        v2 = hy*ev[:,n]
        @views @inbounds for m in eachindex(en)
            en[m]<mu && continue
            hnm= dot(v1,ev[:,m])*dot(ev[:,m],v2)
            dE = en[n]-en[m]
            Threads.@threads for iw in eachindex(w)
                Xw[iw]+= hnm/(w[iw]+η+dE) - conj(hnm)/(w[iw]+η-dE)
            end
        end
    end
    return Xw
end

function xcsz(en,ev,jx,jy,sz)
    s_arr = Array{Float64}(undef,2)
    nth = Threads.nthreads()
    sm_arr = Array{Float64}(undef,nth)
    chs = chunks(eachindex(en), n=nth)
    @views for n in 1:2
        v1 = sz*ev[:,n]
        v2 = jy*ev[:,n]
        Threads.@threads for i in 1:nth
            sm = 0.0im
            for m in chs[i]
                m == n && continue
                dE = abs(en[m]-en[n])<1e-6 ? 1e-6 : en[m]-en[n]
                sm+=dot(v1,ev[:,m])*dot(ev[:,m],v2)/dE^2
            end
            sm_arr[i] = 2*real(-1im*sm)
        end
        s_arr[n] = sum(sm_arr)*real(dot(ev[:,n],jx,ev[:,n]))
    end
    return s_arr
end

function cal_tau0(en,ev,jx,tau)
    s_arr = Array{ComplexF64}(undef,2)
    nth = Threads.nthreads()
    chs = chunks(eachindex(en), n=nth)
    @views for n in 1:2
        v1 = jx*ev[:,n]
        v2 = tau*ev[:,n]
        sm_arr = Array{ComplexF64}(undef,nth)
        Threads.@threads for i in 1:nth
            sm = 0.0im
            for m in chs[i]
                m == n && continue
                dE = abs(en[m]-en[n])<1e-6 ? 1e-6 : en[m]-en[n]
                sm+=dot(v1,ev[:,m])*dot(ev[:,m],v2)/dE
            end
            sm_arr[i] = sm
        end
        s_arr[n] = sum(sm_arr)*1im
    end
    return s_arr
end

function cal_tau1(en,ev,jx,jy,tau)
    s_arr = Array{Float64}(undef,2)
    nth = Threads.nthreads()
    chs = chunks(eachindex(en),n=nth)
    @views for n in 1:2
        v1 = jx*ev[:,n]
        v3 = jy*ev[:,n]
        sm=0.0
        for m in eachindex(en)
            m == n && continue
            dmn = abs(en[n]-en[m])
            dmn<1e-5 && (dmn=1e-5)

            v2=tau*ev[:,m]
            hmn = dot(ev[:,m],v3)/dmn^2
            Threads.@threads for i in 1:nth
                sl = 0.0im
                sl_arr = Array{ComplexF64}(undef,nth)
                for l in chs[i]
                    l == n && continue
                    dE = abs(en[l]-en[n])
                    dE<1e-6 && (dE=1e-6)
                    sl+=dot(v1,ev[:,l])*dot(ev[:,l],v2)/dE
                end
                sl_arr[i] = sl
            end
            sm+=2*real(sum(sl_arr)*hmn)
        end
        s_arr[n]=sm
    end
    return s_arr
end

function cal_tau2(en,ev,jx,jy,tau)
    s_arr = Array{Float64}(undef,2)
    nth = Threads.nthreads()
    chs = chunks(eachindex(en), n=nth)
    @views for n in 1:2
        v1 = jx*ev[:,n]
        v2 = jy*ev[:,n]
        Threads.@threads for i in 1:nth
            sm = 0.0im
            sm_arr = Array{ComplexF64}(undef,nth)
            for m in chs[i]
                m == n && continue
                dE = abs(en[m]-en[n])<1e-5 ? 1e-5 : en[m]-en[n]
                sm+=dot(v1,ev[:,m])*dot(ev[:,m],v2)/dE^3
            end
            sm_arr[i] = 2*real(sm)
        end
        s_arr[n]=real(dot(ev[:,n],tau,ev[:,n]))*sum(sm_arr)
    end
    return s_arr
end

function FermiHall(bz,lat,Nb,mu)
    (;v0,m0,mz,NK,Kvec,Kcoe) = lat
    mat=zeros(ComplexF64,2*NK,2*NK)
    matoff!(mat,v0,m0,Kcoe)

    _,Nx,Ny = size(bz)
    s1 = Array{Float64}(undef,Nb,Nx,Ny)
    s2 = similar(s1)
    sz = similar(s1)
    lmz = abs(mz)<1e-9
    for iy in 1:Ny,ix in 1:Nx
        matdiag!(mat,bz[:,ix,iy],Kvec, v0, mz)
        en,ev=eigen(Hermitian(mat))
        lmz && zeeman_split!(ev)

        J1s= cal_Jθ(bz[:,ix,iy],Kvec,0.0;sp=-1)
        J1 = cal_Jθ(bz[:,ix,iy],Kvec,0.0,sp=1)
        J2 = cal_Jθ(bz[:,ix,iy],Kvec,pi/2,sp=1)
        s1[:,ix,iy].= _fermiHall(en,ev,J1s,J2,Nb,mu)
        s2[:,ix,iy].= _fermiBerry(en,ev,J1,J2,Nb,mu)
        for i in 1:Nb
            sz[i,ix,iy].=real(dot_sz(ev[:,i]))
        end
    end
    return s1,s2,sz
end

function FermiHall_mu(bz,lat,Nb,mu)
    (;v0,m0,mz,NK,Kvec,Kcoe) = lat
    mat=zeros(ComplexF64,2*NK,2*NK)
    matoff!(mat,v0,m0,Kcoe)

    _,Nx,Ny = size(bz)
    s1 = Array{Float64}(undef,Nb,Nx,Ny,length(mu))
    s2 = similar(s1)
    # s3 = similar(s1)
    sz = Array{Float64}(undef,Nb,Nx,Ny)
    # en2d = similar(sz)
    lmz = abs(mz)<1e-9
    for iy in 1:Ny,ix in 1:Nx
        matdiag!(mat,bz[:,ix,iy],Kvec, v0, mz)
        en,ev=eigen(Hermitian(mat))
        lmz && zeeman_split!(ev)
        # en2d[:,ix,iy].=en[1:Nb]

        J1s= cal_Jθ(bz[:,ix,iy],Kvec,0.0;sp=-1)
        J1 = cal_Jθ(bz[:,ix,iy],Kvec,0.0,sp=1)
        J2 = cal_Jθ(bz[:,ix,iy],Kvec,pi/2,sp=1)
        for iu in eachindex(mu)
            s1[:,ix,iy,iu].= _fermiHall(en,ev,J1s,J2,Nb,mu[iu])
            s2[:,ix,iy,iu].= _fermiHall(en,ev,J1,J2,Nb,mu[iu])
            # s3[:,ix,iy,iu].= _fermiBerry(en,ev,J1,J2,Nb,mu[iu])
        end
        for i in 1:Nb
            sz[i,ix,iy]=real(dot_sz(ev[:,i]))
        end
    end
    return (;s1,s2,sz)
    # return (;s1,s2,s3,sz,en2d)
end