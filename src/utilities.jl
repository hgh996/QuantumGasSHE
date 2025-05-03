export BzLine,mymesh,norBev!,gaugev!,Schmidt,ArrayCutOff,myfilter,expshow

# export Band,set_band,bz2d,norBev!,gaugev!,hinv,Schmidt,ArrayCutOff,myfilter,expshow

# 设置k∈Bz, 用于计算能带
struct BzLine
    Nk::Int                # k点总数
    pt::Array{Int,1}       # 高对称点位置
    K::Array{Float64,2}    # 高对称点坐标
    k::Array{Float64,2}    # k点坐标
    r::Array{Float64,1}    # 路程
end

""" 
    LinBz([A, B, C...], N)
经过点A,B,C...的折线，划分为N个点
"""
function BzLine(Klist::Vector{Vector{Float64}},Nk::Int)
    ndim=length(Klist[1])
    Nlist=length(Klist)
    @assert Nk>=Nlist "Nk input error"

    K=Array{Float64,2}(undef,ndim,Nlist) # 储存高对称点坐标
    K.=hcat(Klist...)
    
    Npath=Nlist-1
    totalpath=0.0 # 总长度
    path=Vector{Float64}(undef,Npath) # 每段折线的长度

    @views for ii=1:Npath
        path[ii] = norm(K[:,ii+1].-K[:,ii])
        @assert abs(path[ii])>1e-7 "Klist input error"
        totalpath+=path[ii]
    end

    Nk2 = Nk-Nlist                       # 除了高对称点以外的点的数目
    NKpath = Vector{Int}(undef,Npath)    # 每段折线k的点数
    dpath = Vector{Float64}(undef,Npath) # 每段折线细分的长度
    for ii in 1:Npath
        NKpath[ii]=trunc(Int,Nk2*path[ii]/totalpath) # 按折线长度比例划分点
        dpath[ii]=path[ii]/(NKpath[ii]+1)
    end
    for ii=1:Npath
        Nk2-=NKpath[ii] # 按长度划分后剩下的点
    end
    while Nk2>0         # 剩下的点按 dpath 长度继续分配
        _,pmax=findmax(dpath) 
        NKpath[pmax]+=1
        dpath[pmax]=path[pmax]/(NKpath[pmax]+1)
        Nk2-=1
    end
    @assert sum(NKpath)+Nlist==Nk "KK partition error"

    kk=Array{Float64,2}(undef,ndim,Nk)
    ik=0
    for ii=1:Npath
        Nktmp=NKpath[ii]+2
        xrn=[range(K[id,ii],K[id,ii+1],Nktmp) for id in 1:ndim]
        for iv in zip(xrn...)
            ik+=1
            kk[:,ik].=iv
        end
        ik-=1
    end

    rr=Vector{Float64}(undef,Nk) # 路程
    rr[1]=0.0
    ik=1
    for ii=1:Npath
        for _ in 1:NKpath[ii]+1
            rr[ik+1]=rr[ik]+dpath[ii]
            ik+=1
        end
    end

    pt=Vector{Int}(undef,Nlist) # 高对称点点位置
    pt[1]=1
    for ii=1:Npath
        pt[ii+1]=pt[ii]+NKpath[ii]+1
    end

    BzLine(Nk,pt,K,kk,rr)
end

BzLine(Klist::Vector{Vector{Float64}})=BzLine(Klist,length(Klist))

# 网格离散化
""" 
    mymesh([O, A, B], [n1,n2], period)
对矢量OA，OB围成的平行四边形区域进行网格化分，两个方向点的数目为n1，n2
"""
function mymesh(
    plist::Vector{<:AbstractVector{Float64}},
    nn::Vector{Int},
    period::Bool=false
)
    if period
        n1 = nn[1]
        n2 = nn[2]
    else
        n1 = nn[1]-1
        n2 = nn[2]-1
    end
    p=plist[1]
    b1=plist[2]./n1
    b2=plist[3]./n2
    bz=Array{Float64,3}(undef,2,nn[1],nn[2])
    for jj in 1:nn[2],ii in 1:nn[1]
        bz[:,ii,jj].= p.+(jj-1).*b2.+(ii-1).*b1
    end
    return bz
end

# 非均匀网格划分
function mymesh(plist::Vector{<:AbstractVector{Float64}},n1::Vector{Float64},n2::Vector{Float64})
    p=plist[1]
    b1=plist[2]
    b2=plist[3]
    bz=Array{Float64,3}(undef,2,length(n1),length(n2))
    for j in eachindex(n2),i in eachindex(n1)
        bz[:,i,j].= p.+n1[i].*b1.+n2[j].*b2
    end
    return bz
end

# 适合N维情况
function mymesh2(Klist::Vector{<:AbstractVector{Float64}},nn::NTuple{ndim,Int}) where {ndim}
    # ndim=length(nn)
    a=Array{Float64,2}(undef,ndim,ndim)
    p0=Klist[1]
    for ii in 1:ndim
        a[:,ii].=Klist[ii+1]./(nn[ii]-1)
    end

    mgrid=Array{Float64,ndim+1}(undef,ndim,nn...)
    xrn = ntuple(i->1:nn[i], ndim)
    
    mesh_tmp!(mgrid,ndim,xrn,a,p0)
    mgrid
end
function mesh_tmp!(mgrid,ndim,xrn,a,p0)
    Vtmp=Array{Float64}(undef,ndim)
    @inbounds @views for idx in Iterators.product(xrn...)
        Vtmp.= p0 
        for ii in 1:ndim
            Vtmp.+=(idx[ii]-1).*a[:,ii]
        end
        mgrid[:,idx...].=Vtmp
    end
    nothing
end


"""
    norBev!(bv::Matrix; V0_len=1.0)
BdG归一化，其中bv前半部分对应的应为正能量部分。零模无法归一化，只乘以常数V0_len
"""
function norBev!(bv::AbstractMatrix{<:Number}; V0_len::Float64=1.0)
    Nv=size(bv,2)
    lenv=round(Int,size(bv,1)/2)
    for ii=1:Nv
        tmp=0.0
        for jj=1:lenv
            tmp+=abs2(bv[jj,ii])
            tmp-=abs2(bv[jj+lenv,ii])
        end
        tmp=abs(tmp)
        if tmp<1e-6
            for jj=1:2*lenv
                bv[jj,ii]*=V0_len
            end
            continue
        end
        tmp=1.0/sqrt(tmp)
        for jj=1:2*lenv
            bv[jj,ii]*=tmp
        end
    end
    nothing
end

"""
    gaugev!(ev::Array{T})
对数组的每一列取相位规范，其中规范方式为将该列模最大的数取为实数
"""
function gaugev!(ev::Array{T}) where T<:Union{ComplexF64,Float64}
    d=size(ev)
    Nv=1
    if length(d)!=1
        for ii in 2:length(d)
            Nv*=d[ii]
        end
    end

    for ii in 1:Nv
        p1=(ii-1)*d[1]+1
        p2=p1+d[1]-1
        pt=p1
        mpt=abs(ev[pt])+1e-6
        for jj in p1:p2    
            if abs(ev[jj])>mpt
                pt=jj
                mpt=abs(ev[jj])+1e-6
            end
        end
        mpt<2e-6 && continue
        phs::T=abs(ev[pt])/ev[pt]
        for jj in p1:p2
            ev[jj]*=phs
        end
    end
    nothing
end

"""
    hinv(v::Hermitian{T,Matrix{T}}; p::Float64=1e-8)
厄米矩阵求逆，对本征值小于 p 的数的逆取为零
"""
function hinv(v::Hermitian{T,Matrix{T}}; p::Float64=1e-8) where T<:Number
    en,ev=eigen(v)
    en2=similar(en)
    ev2=Array{T,2}(undef,length(en),length(en))
    for ii in eachindex(en)
        if abs(en[ii])>p
            en2[ii]=1/en[ii]
        else
            en2[ii]=0
            # println("hinv error")
            # ev2.=NaN
            # return ev2
        end
    end
    ev2.=ev*Diagonal(en2)*ev'
    return ev2
end

"""
    Schmidt(v::Matrix{T})
斯密特正交化
"""
function Schmidt(v::Matrix{T}) where {T<:Number}
    lv,nv=size(v)
    vo=Matrix{T}(undef,lv,nv)
    vtmp=Array{T}(undef,lv)
    nonzero=0
    for ii=1:nv
        if norm(v[:,ii])>1e-8
            vo[:,ii].=normalize(v[:,ii])
            nonzero=ii
            break
        end
        vo[:,ii].=v[:,ii]
    end
    for ii=nonzero+1:nv
        if norm(v[:,ii])<1e-8
            vo[:,ii].=v[:,ii]
            continue
        end
        vtmp.=v[:,ii]
        for jj=1:ii-1
            vtmp.-=(vo[:,jj]'*v[:,ii]).*vo[:,jj]
        end
        if norm(vtmp)<1e-8
            vo[:,ii].=vtmp
            continue
        end
        vo[:,ii].=normalize(vtmp)
    end
    # nn=Array{Float64,1}(undef,nv)
    # for ii =1:nv
    #     nn[ii]=norm(vo[:,ii])
    # end
    # pt=sortperm(nn,rev=true)
    # vo.=vo[:,pt]
    return vo
end

"""
    sumNaN(a::Array{T})
求和，忽略数组a中的NaN
"""
function sumNaN(a::Array{T}) where {T<:Number}
    tmp=zero(T)
    for ii in a
        isnan(ii) && continue
        tmp+=ii
    end
    return tmp
end

"""
    ArrayCutOff(v::Array{Float64},cutdown::Float64,cutup::Float64)
将数组v的上下界取在cutdown，cutup之间
"""
# 数组截断
function ArrayCutOff(v::Array{Float64},cutdown::Float64,cutup::Float64)
    v1=similar(v)
    for ii in eachindex(v)
        if v[ii]<cutdown
            v1[ii]=cutdown
        elseif v[ii]>cutup
            v1[ii]=cutup
        else
            v1[ii]=v[ii]
        end
    end
    return v1
end

"""
    myfilter(x::Float64,p::Float64=1e-8;digit=4)
对浮点数x进行截断，仅保留4位有效数字，且小于 `1e-8`视为0
"""
function myfilter(x::Float64,p::Float64=1e-8;digit=4)
    abs(x)<p ? 0.0 : round(x,sigdigits=digit)
end

function myfilter(x::ComplexF64,p::Float64=1e-8;digit=4)
    r1=myfilter(real(x),p;digit=digit)
    r2=myfilter(imag(x),p;digit=digit)
    return complex(r1,r2)
end

function myfilter(v::Array{<:Number},p::Float64=1e-8;digit=4)
    v1=similar(v)
    for ii in eachindex(v)
        v1[ii]=myfilter(v[ii],p;digit=digit)
    end
    return v1
end

"""
    expshow(x::ComplexF64,p::Float64=1e-8;digit=4)->(ρ,θ)
求复数的模和相位，仅保留4位有效数字，且小于 `1e-8`视为0
"""
function expshow(a::ComplexF64,p::Float64=1e-8;digit=4)
    (myfilter(abs(a),p;digit=digit),myfilter(angle(a)/1pi,p;digit=digit))
end

function expshow(a::Array{ComplexF64},p::Float64=1e-8;digit=4)
    b=[expshow(ii,p;digit=digit) for ii in a]
    return b
end


