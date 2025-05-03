using Revise
using SpinHall
using OrderedCollections,FileIO #,MAT
using CairoMakie
set_theme!(;size=(600,400))
c = repeat(Makie.wong_colors(),3)
const cm = 72/2.54

function set_lattice(v0, m0, g′)
    mz = 0.0
    g = [0.26,0.22].*g′
    Kmax = 7
    b = [[1.0,1.0] [-1.0,1.0]]
    Lattice(b,v0,m0,mz,g[1],g[2],Kmax)
end



## ---------------------------------------------------------
#       fig (a), Hall conductivity as function of M_0
#  ---------------------------------------------------------
function spinhall_M0(g0::Float64)
    t=time()
    m0 = [range(0.03,1.73,18); range(1.76,1.99,10); range(2.02,3.2,13)]
    Nm0 = length(m0)
    Xw = Array{ComplexF64}(undef,7,Nm0)
    Mspin = Array{Float64}(undef,3,Nm0)
    Γ = [0.0,0.0]
 
    for ii in eachindex(m0)
        println("--------------------------------------------------")
        println("m0 = ",m0[ii])

        lat = set_lattice(8.0,m0[ii],g0)
        E0,ϕ0=eigenband(lat, Γ, 1:20)
        
        if abs(g0)>1e-5
            if 1.679<m0[ii]<1.93
                Nopt=16; Nstep1=3*10^5; Nstep2=10^4
            else
                Nopt=8; Nstep1=10^5; Nstep2=5000
            end
            gs=[1.0,cispi(0.25), zeros(Nopt-2)...]
            ϕG,u0,xopt=main_opt(E0[1:Nopt], ϕ0[:,1:Nopt], lat;gs=gs, Nstep=Nstep1)
            mat= calmat(lat,Γ)

            ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=Nstep2)
            SpinHall.gaugephi0!(ϕG, ϕ0)
        else
            if m0[ii]<1.87
                ϕG = (ϕ0[:,1].+cispi(0.25).*ϕ0[:,2])./√2
                u0 = E0[1]
            else
                ϕG = ϕ0[:,1] # SpinHall.normalize(ϕ0[:,1].+0.5.*ϕ0[:,2])
                u0 = E0[1]
            end
        end
        
        Mspin[:,ii].= real.(dot_spin(ϕG))
        PTϕG = PTtransform(ϕG)
        Vg=[ϕG PTϕG]
        SpinHall.zeeman_split!(Vg)
        ci = Vg'*ϕG
        ci|>expshow|>println
        
        ϕG1=Vg[:,1].*ci[1]
        ϕG2=Vg[:,2].*ci[2]
        
        Mk0 = cal_BdG(lat,ϕG,u0,Γ)

        Jx1,Dhx1 = cal_Ju(ϕG1,Γ,lat.Kvec; u=1,sp=-1)
        Jy1,Dhy1 = cal_Ju(ϕG1,Γ,lat.Kvec; u=2)
        Jx2,Dhx2 = cal_Ju(ϕG2,Γ,lat.Kvec; u=1,sp=-1)
        Jy2,Dhy2 = cal_Ju(ϕG2,Γ,lat.Kvec; u=2)

        Xw[1,ii] = Green1(Mk0,Jx1,Jy1)/lat.Sunit    # ⟨1...1⟩
        Xw[2,ii] = Green1(Mk0,Jx2,Jy2)/lat.Sunit    # ⟨2...2⟩
        Xw[3,ii] = Green1(Mk0,Jx1,Jy2)/lat.Sunit    # ⟨1...2⟩
        Xw[4,ii] = Green1(Mk0,Jx2,Jy1)/lat.Sunit    # ⟨2...1⟩

        Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=-1)
        Jy,Dhy = cal_Ju(ϕG,Γ,lat.Kvec; u=2)
        Xw[5,ii] = Green1(Mk0,Jx,Jy)/lat.Sunit      # σ^s

        tmp = -1.0.*cal_Bcav(lat,Γ,1:2)./lat.Sunit
        Xw[6,ii] = dot_sz(ϕG1)*tmp[1]+dot_sz(ϕG2)*tmp[2] # ⟨ϕ₁⟩B₁+⟨ϕ₂⟩B₂

        ben,bev=eigBdG(Mk0)
        Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=1)
        Xw[7,ii] = Xspec2(Dhx,Dhy,ben,bev,ϕG,PTϕG)/lat.Sunit
    end

    println("time_used: ",time()-t,"\n\n")
    return (;m0,Xw,Mspin)
end

# @time g0=spinhall_M0(0.0); 
@time g1=spinhall_M0(1.0); 

series(g1.m0,[abs.(g1.Mspin[3,:]) sqrt.(g1.Mspin[1,:].^2 .+g1.Mspin[2,:].^2)]'./2,color=c,markersize=10)


## --------------------------------------------------------
#           fig (b),    the spin vecter, Ground state 
# ---------------------------------------------------------
lat = set_lattice(8.0,1.5,1.0)
Γ = [0.0,0.0]
Nopt = 10
E0,ϕ0=eigenband(lat, Γ, 1:Nopt)
init_gs=[1,cispi(-0.25), zeros(Nopt-2)...]

ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=init_gs, Nstep=10^5)
mat = calmat(lat, Γ)
@time ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=10^4)
SpinHall.gaugephi0!(ϕG, ϕ0)
ϕ0'*ϕG|>expshow



## ----  plot ground state ----
x = range(-1.1pi,1.1pi,64)
y = range(-1.1pi,1.1pi,64)

up = cal_bloch_wave(Γ,ϕG[1:lat.NK],lat,x,y)
dn = cal_bloch_wave(Γ,ϕG[lat.NK+1:end],lat,x,y)
sp = cal_bloch_spin(Γ, ϕG, lat, x, x)


##

fig,_,hm = heatmap(abs2.(up),figure=(size=(500,380),),colormap=:jet,axis=(aspect=1,))
Colorbar(fig[1,2],hm)
fig

##
#sp = sp4
nsp= vec(sqrt.(sp[1].^2+sp[2].^2))
arrows(x, x, sp[1], sp[2], arrowsize = 5, lengthscale = 1,
    arrowcolor = nsp, linecolor = nsp, axis=(;aspect=1)
)




# ---------------------------------------------------
##         fig(c)， BdG能谱
# ---------------------------------------------------

kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],128)
xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
@time ben = eig_BdG(lat,ϕG,u0,kl.k,12); 
fig=series(kl.r, ben[1:12,:];
    #color=repeat(Makie.wong_colors(),3),
    solid_color = :blue,
    figure=(size=(1,0.8).*600,),
    linewidth=1.5,
    axis=(;xticks=xt,yticks=range(0,10,6),ygridvisible=false)
)



## ----- supplimentary material figure Spin Hall -----
Mk0 = cal_BdG(lat,ϕG,u0,Γ)
Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=-1)
Jy,Dhy = cal_Ju(ϕG,Γ,lat.Kvec; u=2,sp=1)

w = [range(0,1.5,100); range(1.6,4.4,18); range(4.45,6.0,120)]
Xw1 = Green1(Mk0,w,Jx,Jy,η=0.0)./lat.Sunit
fig = series(w,hcat(reim(Xw1)...)',marker=:circle,axis=(;limits=(nothing,(-0.08,0.08))))


## --- 谱分解计算 Spin Hall ---
ben,bev=eigBdG(Mk0)
Xw2 = Xspec1(w,Dhx,Dhy,ben,bev,ϕG)./lat.Sunit
series!(w,hcat(reim(Xw2)...)',solid_color=:red,linestyle=:dash)
fig


# ---------------------------------------------------
##         fig(d)， 任意角度 spin hall
# ---------------------------------------------------
function hall_theta(lat,ϕG,u0,Γ,N)
    Mk0,_ = cal_BdG(lat,ϕG,u0,Γ)
    Xw = Array{ComplexF64}(undef,2,N)
    θ = range(0,1pi,N)
    w = [0.0,0.8] #[range(0,1.5,20); range(1.6,3,10)]#; range(4.45,6.0,50)]
    for i in 1:N
        J1s = cal_Jθ(ϕG,Γ,lat.Kvec,θ[i]+pi/2; sp=-1)
        J2 = cal_Jθ(ϕG,Γ,lat.Kvec,θ[i]; sp=1)
        J2s = cal_Jθ(ϕG,Γ,lat.Kvec,θ[i]; sp=-1)
    
        Xw1 = Green1(Mk0,w,J1s,J2)./lat.Sunit
        fig = Figure(size=(400,700))

        str = myfilter(θ[i]/pi)
        series(fig[1,1],w,hcat(reim(Xw1)...)',marker=:circle,axis=(limits=(0,3,-0.1,0.1),
                title=L"\sigma_{xy}^s,\quad \theta=%$(str)\pi")
        )

        Xw2 = Green1(Mk0,w,J2s,J2)./lat.Sunit
        series(fig[2,1],w,hcat(reim(Xw2)...)',marker=:circle,axis=(limits=(0,3,-0.1,0.1),
                title=L"\sigma_{yy}^s,\quad \theta=%$(str)\pi")
        )
        display(fig)
        Xw[1,i]=Xw1[1]
        Xw[2,i]=Xw2[1]
    end
    return (;θ, Xw)
end
Xθ1 = hall_theta(lat,ϕG,u0,Γ,65)



##
xt = ([range(0,2pi,9);],[L"%$(i)\pi/4" for i in 0:8])
fig2,_,_=series(Xθ1.θ,real.(Xθ1.Xw),marker=:circle, axis=(xticks=xt,yticks=([-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01])),labels=[L"v_0=8,m_0=1.5,g_{11}=%$(lat.g1)",L"\sigma_{\theta,\theta}^s"],solid_color=:blue)
# series!(Xθ1.θ,real.(Xθ2.Xw),marker=:circle,labels=[L"v_0=4,m_0=0.8,g_{11}=0.26",L"\sigma_{\theta,\theta}^s"],solid_color=:red)
# series!(Xθ1.θ,real.(Xθ3.Xw),marker=:circle,labels=[L"0.2",L"\sigma_{\theta,\theta}^s"],solid_color=:green)
# series!(Xθ1.θ,real.(Xθ4.Xw),marker=:circle,labels=[L"0.1",L"\sigma_{\theta,\theta}^s"],solid_color=:black)
# series!(Xθ1.θ,real.(Xθ5.Xw),marker=:circle,labels=[L"0.05",L"\sigma_{\theta,\theta}^s"],solid_color=:purple)
axislegend()
fig2

# save("data/anisotropy.h5",OrderedDict("Xtheta"=>real.(Xθ.Xw),"theta"=>collect(Xθ.θ)))


## ----------------------------------------------------
#       save data of Bonsonic related calculation
#  ----------------------------------------------------
save("data/Boson.h5", OrderedDict(
    # fig (a)
    "a_m0"    => g1.m0,
    "a_Mspin" => g1.Mspin,
    "a_Xw"    => real.(g1.Xw),  # 补充材料

    # fig (b)
    "b_x"     => collect(x),
    "b_sp1"   => cat(sp1[1],sp1[2],dims=3),
    "b_sp2"   => cat(sp2[1],sp2[2],dims=3), # 补充材料
    "b_sp3"   => cat(sp3[1],sp3[2],dims=3),
    "b_sp4"   => cat(sp4[1],sp4[2],dims=3),

    # fig (c)
    "c_r"     => kl.r,
    "c_pt"    => kl.pt,
    "c_ben"   => ben,

    # 补充材料
    "sm_w"    => w,
    "sm_Xw"   => real.(Xw1),

    # fig (d)
    "d_Xtheta"=> real.(Xθ1.Xw),
    "d_theta" => collect(Xθ1.θ)
))

##
X = load("data/Boson2.h5")
X["c_k"] = kl.k
save("data/Boson2.h5",X)
delete!(X,"C_k")

# matread("data/Boson.mat")