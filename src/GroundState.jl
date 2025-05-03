export main_opt,imag_time_evl,energy,energyint

# 通过数值积分计算四个布洛赫函数的积分值 U_{ijmn} = ∫(ϕᵢϕⱼ)^*ϕₘϕₙ
function phi4_integrate!(
    Uphi4::Array{ComplexF64,4},
    uk::Array{ComplexF64,2},
    lat::Lattice
)
    Nb=size(uk,2)
    M = -0.5.*(lat.a[:,1].+lat.a[:,2])
    bz=mymesh([M,lat.a[:,1],lat.a[:,2]],[160,160]) # 元胞进行网格化分
    _,Nx,Ny=size(bz)

    xx=range(0.0,norm(lat.a[:,1]),Nx)   # 数值积分变量 x,y
    yy=range(0.0,norm(lat.a[:,2]),Ny)

    ψup=cal_bloch_wave([0.0,0.0],uk[1:lat.NK,:],lat,bz)
    ψdw=cal_bloch_wave([0.0,0.0],uk[lat.NK+1:end,:],lat,bz)
    (g1,g2) = (lat.g1,lat.g2).*lat.Sunit

    print("check_int:") # 检查积分的归一化情况
    println(trapz((xx,yy),abs2.(ψup[:,:,1]))+trapz((xx,yy),abs2.(ψdw[:,:,1])))

    Threads.@threads for jj in 1:Nb
        tmp=Array{ComplexF64,2}(undef,Nx,Ny)
        uu1=Array{ComplexF64,2}(undef,Nx,Ny)
        dd1=Array{ComplexF64,2}(undef,Nx,Ny)
        ud1=Array{ComplexF64,2}(undef,Nx,Ny)
        du1=Array{ComplexF64,2}(undef,Nx,Ny)
        uu2=Array{ComplexF64,2}(undef,Nx,Ny)
        dd2=Array{ComplexF64,2}(undef,Nx,Ny)
        ud2=Array{ComplexF64,2}(undef,Nx,Ny)
        du2=Array{ComplexF64,2}(undef,Nx,Ny)
        @inbounds for ii in 1:Nb
            for iy in 1:Ny,ix in 1:Nx
                uu1[ix,iy]=conj(ψup[ix,iy,ii]*ψup[ix,iy,jj])
                dd1[ix,iy]=conj(ψdw[ix,iy,ii]*ψdw[ix,iy,jj])
                ud1[ix,iy]=conj(ψup[ix,iy,ii]*ψdw[ix,iy,jj])
                du1[ix,iy]=conj(ψdw[ix,iy,ii]*ψup[ix,iy,jj])
            end
            for nn in 1:Nb
                for iy in 1:Ny,ix in 1:Nx
                    uu2[ix,iy]=uu1[ix,iy]*ψup[ix,iy,nn]
                    dd2[ix,iy]=dd1[ix,iy]*ψdw[ix,iy,nn]
                    ud2[ix,iy]=ud1[ix,iy]*ψdw[ix,iy,nn]
                    du2[ix,iy]=du1[ix,iy]*ψup[ix,iy,nn]
                end
                for mm in 1:Nb
                    for iy in 1:Ny,ix in 1:Nx
                        tmp[ix,iy]=uu2[ix,iy]*ψup[ix,iy,mm]
                    end
                    Uphi4[mm,nn,ii,jj]=trapz((xx,yy),tmp)*g1

                    for iy in 1:Ny,ix in 1:Nx
                        tmp[ix,iy]=dd2[ix,iy]*ψdw[ix,iy,mm]
                    end
                    Uphi4[mm,nn,ii,jj]+=trapz((xx,yy),tmp)*g1

                    for iy in 1:Ny,ix in 1:Nx
                        tmp[ix,iy]=ud2[ix,iy]*ψup[ix,iy,mm]
                    end
                    Uphi4[mm,nn,ii,jj]+=trapz((xx,yy),tmp)*g2

                    for iy in 1:Ny,ix in 1:Nx
                        tmp[ix,iy]=du2[ix,iy]*ψdw[ix,iy,mm]
                    end
                    Uphi4[mm,nn,ii,jj]+=trapz((xx,yy),tmp)*g2
                end
            end
        end
    end
    nothing
end

# (ϕᵢϕⱼ)^*ϕₘϕₙ 展开为平面波，不为零的项的指标
function phi4_coe(Kcoe::Array{Int,2})
    NK=size(Kcoe,2)
    tmp=Array{Int,2}(undef,4,NK^3*2)
    kk=0
    @inbounds for ii in 1:NK, jj in 1:NK, mm in 1:NK
        t1=Kcoe[1,mm]-Kcoe[1,ii]-Kcoe[1,jj]
        t2=Kcoe[2,mm]-Kcoe[2,ii]-Kcoe[2,jj]
        for nn in 1:NK
            if t1+Kcoe[1,nn]==0 && t2+Kcoe[2,nn]==0
                kk+=1
                tmp[:,kk].=ii,jj,mm,nn
                break
            end
        end
    end
    println("NKl: ",kk)
    kk>NK^3*2 && println("NKl error")
    return tmp[:,1:kk]
end

function cal_phi4_coe(Kcoe::Array{Int,2},Kmax::Int)
    local Kl::Array{Int,2}
    if isfile("data/Kl"*string(Kmax)*".jld2")
        Kl=load("data/Kl"*string(Kmax)*".jld2","Kl")
    else
        Kl=phi4_coe(Kcoe)
        save("data/Kl"*string(Kmax)*".jld2","Kl",Kl)
    end
    return Kl
end

# (ϕᵢϕⱼ)^*ϕₘϕₙ 展开为平面波，对不为零的项求和得到积分值
function phi4_integrate2!(
    Uphi4::Array{ComplexF64,4},
    uk::Array{ComplexF64,2},
    lat::Lattice
)
    Nb=size(uk,2)
    (;g1,g2,NK) = lat

    Kl=cal_phi4_coe(lat.Kcoe,lat.Kmax)
    NKl=size(Kl,2)

    Threads.@threads for jj in 1:Nb
        @inbounds for ii in 1:Nb,nn in 1:Nb,mm in 1:Nb
            uu=ud=dd=0.0im
            for iQ in 1:NKl
                t1,t2,t3,t4=view(Kl,:,iQ)
                uu+=conj(uk[t1,ii]*uk[t2,jj])*uk[t3,mm]*uk[t4,nn]
                ud+=conj(uk[t1,ii]*uk[t2+NK,jj])*uk[t3+NK,mm]*uk[t4,nn]
                ud+=conj(uk[t1+NK,ii]*uk[t2,jj])*uk[t3,mm]*uk[t4+NK,nn]
                dd+=conj(uk[t1+NK,ii]*uk[t2+NK,jj])*uk[t3+NK,mm]*uk[t4+NK,nn]
            end
            Uphi4[mm,nn,ii,jj]=uu*g1+ud*g2+dd*g1
        end
    end

    return nothing
end


# -------------------------------------------
#           数值优化算基态
# -------------------------------------------
function energy(coe::Vector{ComplexF64},E0::Vector{Float64},Uphi4::Array{ComplexF64,4})
    normalize!(coe)
    Nb=length(E0)
    en=0.0
    for ii in 1:Nb
        en+=abs2(coe[ii])*E0[ii]
    end

    Eint=0.0im
    @inbounds for jj in 1:Nb,ii in 1:Nb
        tmp1=conj(coe[ii]*coe[jj])
        for nn in 1:Nb
            tmp2=tmp1*coe[nn]
            for mm in 1:Nb
                Eint+=coe[mm]*tmp2*Uphi4[mm,nn,ii,jj]
            end
        end
    end

    u0=en+real(Eint) # 化学势
    en+=real(Eint)*0.5
    return en,u0
end

function energy!(
    coe::Vector{Float64},       # 实数形式的基态展开系数
    E0::Vector{Float64},        # 无相互作用能量
    Uphi4::Array{ComplexF64,4}, # phi-4 积分
    coe2::Vector{ComplexF64}    # 实数形式的基态展开系数
)
    t0=0.0
    @inbounds for ii in coe
        t0+=abs2(ii)
    end
    t0=1/sqrt(t0)

    Nb=length(E0)
    en=0.0
    @inbounds for ii in 1:Nb
        coe2[ii]=complex(coe[2*ii-1],coe[2*ii])*t0
        en+=abs2(coe2[ii])*E0[ii]
    end

    Eint=0.0im
    @inbounds for jj in 1:Nb,ii in 1:Nb
        tmp1=conj(coe2[ii]*coe2[jj])
        for nn in 1:Nb
            tmp2=tmp1*coe2[nn]
            for mm in 1:Nb
                Eint+=coe2[mm]*tmp2*Uphi4[mm,nn,ii,jj]
            end
        end
    end

    en+=real(Eint)*0.5
    return en
end

function optmin(
    gs::Vector{Float64},
    Nstep::Int,
    en::Vector{Float64},
    Uphi4::Array{ComplexF64,4}
)
    Nb = length(en)
    coe = Array{ComplexF64}(undef,Nb)
    foo2(cc::Vector{Float64}) = energy!(cc,en,Uphi4,coe)
    res=bboptimize(foo2,gs;
        SearchRange=(-1.0,1.0),
        NumDimensions=2*Nb,MaxFuncEvals=Nstep,TraceMode=:silent
    )
    gaugev!(coe)
    opt_en=best_fitness(res)
    return coe,opt_en
end

function Vcomplex_to_Vreal(x::AbstractArray{ComplexF64})
    y = Array{Float64}(undef,2*length(x))
    i = 1
    for ix in x
        y[i] = real(ix)
        y[i+1] = imag(ix)
        i+=2
    end
    y
end

"""
    main_opt(en,ϕ,lat; gs=rand(ComplexF64,length(en)), Nstep=100000, showres=true)
数值优化求基态
"""
function main_opt(
    en::Vector{Float64},    # 单粒子能量
    ϕ::Array{ComplexF64,2}, # 布洛赫态
    lat::Lattice;
    gs::Vector{ComplexF64}=rand(ComplexF64,length(en)),    # 初始值
    Nstep::Int=100000,      # 优化步数
    showres::Bool=true
)
    Nb=size(ϕ,2)
    Uphi4=Array{ComplexF64,4}(undef,Nb,Nb,Nb,Nb)
    phi4_integrate!(Uphi4,ϕ,lat)

    coe,_ = optmin(Vcomplex_to_Vreal(gs),Nstep,en,Uphi4)
    en0,u0 = energy(coe,en,Uphi4)

    if showres
        println("opt_energy: ",en0)
        display(expshow(coe))
    end

    ϕG=zeros(ComplexF64,2*lat.NK)
    for ii in 1:Nb,iQ in 1:2*lat.NK
        ϕG[iQ]+=ϕ[iQ,ii]*coe[ii]
    end

    return ϕG,u0,coe
end



# -------------------------------------------
#       虚时演化算基态
# -------------------------------------------
function phi4_coe1(Kcoe::Array{Int,2})
    NK=size(Kcoe,2)
    Kl=Array{Int,2}(undef,3,NK^3*2)
    Klp=Array{Int,1}(undef,NK+1)
    iQ=0
    @inbounds for ii in 1:NK
        Klp[ii]=iQ+1
        for jj in 1:NK,mm in 1:NK
            t1=Kcoe[1,mm]-Kcoe[1,ii]-Kcoe[1,jj]
            t2=Kcoe[2,mm]-Kcoe[2,ii]-Kcoe[2,jj]
            for nn in 1:NK
                if t1+Kcoe[1,nn]==0 && t2+Kcoe[2,nn]==0
                    iQ+=1
                    Kl[1,iQ]=jj
                    Kl[2,iQ]=mm
                    Kl[3,iQ]=nn
                    break
                end
            end
        end
    end
    Klp[end]=iQ+1
    return Klp,Kl[:,1:iQ]
end


function energyint(uk::Vector{ComplexF64},lat::Lattice)
    Kl = cal_phi4_coe(lat.Kcoe,lat.Kmax)
    NKl=size(Kl,2)
    uu=ud=dd=0.0im
    NK=lat.NK
    for iQ in 1:NKl
        i,j,k,l=view(Kl,:,iQ)
        uu+=conj(uk[i]*uk[j])*uk[k]*uk[l]
        ud+=conj(uk[i]*uk[j+NK])*uk[k+NK]*uk[l]
        ud+=conj(uk[i+NK]*uk[j])*uk[k]*uk[l+NK]
        dd+=conj(uk[i+NK]*uk[j+NK])*uk[k+NK]*uk[l+NK]
    end
    Eint=(uu*lat.g1+ud*lat.g2+dd*lat.g1)*0.5

    return Eint
end

function time_invlove!(
    Klp::Array{Int,1},
    Kl::Array{Int,2},
    ϕ1::Vector{ComplexF64},
    ϕ2::Vector{ComplexF64},
    g1::Float64,
    g2::Float64,
    hk::Matrix{ComplexF64},
    NQ::Int,
    dt::Float64
)
    Threads.@threads for iQ in 1:NQ
        ψup1=ψup2=ψdw1=ψdw2=0.0im
        @inbounds for ii in Klp[iQ]:Klp[iQ+1]-1
            jj=Kl[1,ii]
            ll=Kl[2,ii]
            mm=Kl[3,ii]
            ψup1+=conj(ϕ1[jj])*ϕ1[ll]*ϕ1[mm]
            ψup2+=conj(ϕ1[jj+NQ])*ϕ1[ll+NQ]*ϕ1[mm]
            ψdw1+=conj(ϕ1[jj+NQ])*ϕ1[ll+NQ]*ϕ1[mm+NQ]
            ψdw2+=conj(ϕ1[jj])*ϕ1[ll]*ϕ1[mm+NQ]
        end
        ψup1=g1*ψup1+g2*ψup2
        ψdw1=g1*ψdw1+g2*ψdw2
        @inbounds for ii in 1:2*NQ
            ψup1+=hk[iQ,ii]*ϕ1[ii]
            ψdw1+=hk[iQ+NQ,ii]*ϕ1[ii]
        end
        ϕ2[iQ]=ψup1
        ϕ2[iQ+NQ]=ψdw1
    end
    Threads.@threads for iQ in 1:2*NQ
        @inbounds ϕ1[iQ]+=ϕ2[iQ]*dt
    end
    nothing
end

"""
    imag_time_evl(hk,ϕ,lat; Nstep=5000,dt=-0.002,showres=true)
虚时演化
"""
function imag_time_evl(
    hk::Matrix{ComplexF64},
    ϕ::Vector{ComplexF64},
    lat::Lattice;
    Nstep::Int=5000,
    dt::Float64=-0.002,
    showres::Bool=true
)
    Klp,Kl=phi4_coe1(lat.Kcoe)
    ϕ1=copy(ϕ)
    normalize!(ϕ1)
    ϕ2=similar(ϕ1)

    for _ in 1:Nstep
        time_invlove!(Klp,Kl,ϕ1,ϕ2,lat.g1,lat.g2,hk,lat.NK,dt)
        normalize!(ϕ1)
    end

    Eint=energyint(ϕ1,lat)
    E0=dot(ϕ1,hk,ϕ1)
    if showres
        println("img_evl_res: ",real(E0+Eint))
        println("initial_ratio: ",abs(dot(ϕ1,ϕ)))
    end

    return ϕ1,real(E0+Eint*2)
end

# 基态单粒子态展开，占比最大的系数取为实数
function gaugephi0!(ϕG::Vector{ComplexF64},ϕ0::Array{ComplexF64,2})
    t1=dot(ϕ0[:,1],ϕG)
    t2=dot(ϕ0[:,2],ϕG)
    abs(t1)+1e-6>abs(t2) ? phs=abs(t1)/t1 : phs=abs(t2)/t2
    for ii in eachindex(ϕG)
        ϕG[ii]*=phs
    end
    nothing
end
