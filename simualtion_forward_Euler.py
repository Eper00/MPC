import model_dummy as m
import numpy as np
import matplotlib.pyplot as plt
Model=m.Transition_model(m.population-m.Latent,m.Latent,m.parameters)
u=0.0
dt=0.5
days=365
step= (int)(days/dt)
hospital=np.zeros(step,float)

T=np.linspace(0,days,step)

for t in range(step):
    dS=Model.dSdt(u)*dt
    dL=Model.dLdt(u)*dt
    dP=Model.dPdt()*dt 
    dA=Model.dAdt()*dt  
    dI=Model.dIdt()*dt
    dR=Model.dRdt()*dt
    dH=Model.dHdt()*dt
    dD=Model.dDdt()*dt
    
    hospital[t]=Model.H
    
    Model.S=Model.S+dS
    Model.L=Model.L+dL
    Model.P=Model.P+dP 
    Model.A=Model.A+dA  
    Model.I=Model.I+dI
    Model.R=Model.R+dR
    Model.H=Model.H+dH
    Model.D=Model.D+dD
    
    
    

plt.plot(T,hospital)
plt.grid()
plt.show()
