# ---------------------------------------------------
##              spin hall of fermi gases
# ---------------------------------------------------
using Revise
using SpinHall
using OrderedCollections,FileIO
using CairoMakie
set_theme!(;size=(600,400))
c = repeat(Makie.wong_colors(),3)
const cm = 72/2.54

function set_lattice(v0, m0, g′, Kmax::Int=7)
    mz = 0.00
    g = [0.26,0.22].*g′
    b = [[1.0,1.0] [-1.0,1.0]]
    Lattice(b,v0,m0,mz,g[1],g[2],Kmax)
end


## --- noninteracting bloch band --
Γ = [0.0,0.0]
lat = set_lattice(4.0,1.5,1.0)
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],256)
en = eigband(lat,kl.k, 1:12)
xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
series(kl.r,en; axis=(;xticks=xt),color=c)



## --- spin hall as function of M_0 ---
function cal_mu_M0(M0,v0,bz,Nb)
    mu = Array{Float64}(undef,3)
    
    lat = set_lattice(v0,M0,0.0,6)
    en = eigband(lat,bz,1:Nb)
    mu[1] = maximum(en[Nb,:,:])+1e-12

    Dos = sort(reshape(en[Nb,:,:],:))
    Ne = length(Dos)
    mid=div(Ne,2)
    mu[2] = iseven(Ne) ? (Dos[mid]+Dos[mid+1])/2 : Dos[mid+1]+1e-13
    mu[3] = minimum(en[Nb,:,:])-1e-12

    return mu
end

function Fermi_M0(M0,bz,ktmp)
    Nb = 4
    v0=4.0
    s_M0 = Array{Float64}(undef,4,length(M0))
    mu=Array{Float64}(undef,3,length(M0))
    t=time()
    for im in eachindex(M0)
        print("$(im)/$(length(M0)),")
        mu[:,im].= cal_mu_M0(M0[im],v0,bz,Nb)
        lat = set_lattice(v0,M0[im],0.0)
        x = SpinHall.FermiHall_mu(bz,lat,Nb,mu[1:2,im])

        fig = Figure(size=(800,720),title=L"m_0=%$(M0[im])")
        str="=$(myfilter(mu[1,im]))"
        _,hm1 = heatmap(fig[1,1],ktmp,ktmp,x.s1[1,:,:,1],axis=(aspect=1,title=L"n=1,\mu=%$(str)"))
        Colorbar(fig[1,2], hm1)
        _,hm2 = heatmap(fig[1,3],ktmp,ktmp,x.s1[3,:,:,1],axis=(aspect=1,title=L"n=3,\mu%$(str)"))
        Colorbar(fig[1,4], hm2)
        str="=$(myfilter(mu[2,im]))"
        _,hm3 = heatmap(fig[2,1],ktmp,ktmp,x.s1[1,:,:,2],axis=(aspect=1,title=L"n=1,\mu%$(str)"))
        Colorbar(fig[2,2], hm3)
        _,hm4 = heatmap(fig[2,3],ktmp,ktmp,x.s1[3,:,:,2],axis=(aspect=1,title=L"n=3,\mu%$(str)"))
        Colorbar(fig[2,4], hm4)
        display(fig)
       
        stmp = zeros(length(ktmp),length(ktmp))
        for iu in 1:2
            stmp.= 0.0
            for i in 1:Nb
                stmp.+=x.s1[i,:,:,iu]
            end
            s_M0[iu,im] = trapz((ktmp,ktmp),stmp)/lat.Sunit
        end

        for iu in 1:2
            stmp.= 0.0
            for i in 1:Nb
                stmp.+=x.s2[i,:,:,iu].*x.sz[i,:,:]
            end
            s_M0[iu+2,im] = trapz((ktmp,ktmp),stmp)/lat.Sunit
        end
        println("time_used:",time()-t)
    end
    return (;s_M0, mu)
end


##
bz = mymesh([(lat.b[:,1].+lat.b[:,2])./(-2),lat.b[:,1],lat.b[:,2]],[256,256])
ktmp=range(0,1,size(bz,3))
scatter(reshape(bz,2,:),axis=(aspect=1,),markersize=2,figure=(size=(600,600),))|>display
M0 = range(0,2,21)
s_M0,mu = Fermi_M0(M0,bz,ktmp)


save("data/Fermi_M0.h5",
    OrderedDict("M0"=>collect(M0),
                "s_M0"=>s_M0,
                "mu"=>mu
    )
)



## ---- 画图 -----

M0 = load("data/Fermi_M0.h5","M0")
s_M0 = load("data/Fermi_M0.h5","s_M0")

##
ib = 1
title=["insulator","metal"]
fig=Figure(size=(1,1.6).*400)
ax = Axis(fig[1,1],title=L"%$(title[ib]),$V_0=%$(lat.v0)$",xlabel=L"M_0/E_r")
scatterlines!(M0,s_M0[ib,:],label=L"\sigma_{xy}^s",marker=:circle)
scatterlines!(M0,s_M0[ib+2,:],label=L"\langle \sigma_z\rangle B'",marker=:utriangle,linestyle=:dash)
axislegend(position=:lt)

ib=2
ax = Axis(fig[2,1],title=L"%$(title[ib]),$V_0=%$(lat.v0)$",xlabel=L"M_0/E_r")
scatterlines!(M0,s_M0[ib,:],label=L"\sigma_{xy}^s",marker=:circle)
scatterlines!(M0,s_M0[ib+2,:],label=L"\langle \sigma_z\rangle B'",marker=:utriangle,linestyle=:dash)
axislegend(position=:lt)
fig