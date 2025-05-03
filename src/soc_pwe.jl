using MKL
using LinearAlgebra
using Plots


# --------------------------------------------
#       计算平面波展开的基矢Q
# --------------------------------------------
function CalQ(Qcut::Int,b1::Vector{Float64},b2::Vector{Float64})
    p = Qcut*max(norm(b1),norm(b2))+1e-8    # 截断半径
    Qc=Array{Int,2}(undef,2,(2*Qcut)^2)     # 用于存储 Q=m*b1+n*b2的系数m,n
    Q=Array{Float64,2}(undef,2,(2*Qcut)^2)  # 用于存储 Q=m*b1+n*b2这个矢量Q
    
    idx=0 # 计数，用来计算 Q 的数量
    for n in -Qcut:Qcut, m in -Qcut:Qcut
        tmp=(m*b1[1]+n*b2[1])^2+(m*b1[2]+n*b2[2])^2
        if sqrt(tmp)<p
            idx+=1
            Qc[:,idx].=m,n
            Q[:,idx].=m.*b1.+n.*b2
        end
    end
    return Qc[:,1:idx],Q[:,1:idx]
end

b1=[1.0,1.0]
b2=[-1.0,1.0]
Qcut=6 
Qc,Q=CalQ(Qcut,b1,b2)



# --------------------------------------------
#       计算矩阵 H_{↓↑}
# --------------------------------------------
function CalH21(Qc::Array{Int,2},M0::Float64)
    NQ=size(Qc,2)   # 算出Q的数量
    H21=zeros(ComplexF64,NQ,NQ) 
    for jj in 1:NQ,ii in 1:NQ
        m=Qc[1,ii]-Qc[1,jj]
        n=Qc[2,ii]-Qc[2,jj]
        if m==1 && n==0
            H21[ii,jj]=(1-1im)*M0/4
        elseif m==0 && n==-1
            H21[ii,jj]=-(1+1im)*M0/4
        elseif m==0 && n==1
            H21[ii,jj]=(1+1im)*M0/4
        elseif m==-1 && n==0
            H21[ii,jj]=-(1-1im)*M0/4
        end
    end
    return H21
end

# --------------------------------------------
#       计算矩阵 H_{↑↑} 的非对角元
# --------------------------------------------
function CalH11(Qc::Array{Int,2},V0::Float64)
    NQ=size(Qc,2)
    H11=zeros(ComplexF64,NQ,NQ)
    for jj in 1:NQ,ii in 1:NQ
        m=Qc[1,ii]-Qc[1,jj]
        n=Qc[2,ii]-Qc[2,jj]
        if (m==1&&n==-1)||(m==-1&&n==1)||(m==1&&n==1)||(m==-1&&n==-1)
            H11[ii,jj]=V0/4
        end
    end
    return H11
end

V0=4.0
M0=2.0
H11=CalH11(Qc,V0)
H21=CalH21(Qc,M0)
H=[H11 H21';H21 H11] # 合并为总的哈密顿矩阵




# --------------------------------------------
#       计算k空间(-1,0)到(1,0)的能带
# --------------------------------------------
function eigHk(H,Nb,Q,V0)
    NQ=size(Q,2)
    ky=0.0
    kx=LinRange(-1,1,101)
    Nk=length(kx)

    en=Array{Float64,2}(undef,Nb,Nk)
    ev=Array{ComplexF64,3}(undef,2*NQ,Nb,Nk)
    for ik in 1:Nk
        for ii in 1:NQ
            H[ii,ii]=(kx[ik]+Q[1,ii])^2+(ky+Q[2,ii])^2+V0
            H[ii+NQ,ii+NQ]=H[ii,ii]     # H22对角元
        end
        entmp,evtmp=eigen(Hermitian(H)) # 对角化哈密顿量
        pt=partialsortperm(entmp,1:Nb)  # 排序，找出前 Nb 条带的位置
        en[:,ik].=entmp[pt]             # 取出前 Nb 条带
        ev[:,:,ik].=evtmp[:,pt]         # 取出对应的本征态
    end
    return en,ev
end

# 计算自旋σ_z的平均值
function meanz(ev)
    Nv=round(Int,length(ev)/2)
    sz=0.0
    for ii in 1:Nv
        sz+=abs2(ev[ii])-abs2(ev[ii+Nv])
    end
    return sz
end

Nb=16 # 保存的能带数目
en,ev=eigHk(H,Nb,Q,V0)
plot(LinRange(-1,1,101),en',legend=false,framestyle=:box,size=(400,300))|>display
display(en[:,51]) # 每条带都是二重简并的
meanz(ev[:,1,51])

