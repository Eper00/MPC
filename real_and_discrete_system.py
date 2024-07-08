import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



population=9800000.
Latent=10.
u=0.
y0 = np.array([population-Latent, Latent,0,0,0,0,0,0])  
t0 = 0.
t_end = 365.
t_span=np.array([t0,t_end])
dt = 1
times=np.linspace(t_span[0],t_span[1],1000)

t=np.linspace(t0,t_end,1000)

def realsystem():
    soln=solve_ivp(dydt,t_span,y0,t_eval=times)
    return soln

def dydt(t, y,u=0):
   
                    #0->beta   #1->delta    #2->N          #3->alpha    #4->p    #5->q     #6->ro_1   #7->ro_a  #8->eta  #9->h    #10->mikro
    param=np.array([1/3 ,     0.75 ,       population,    1/2.5 ,      1/3 ,    0.6 ,      1/4 ,     1/4 ,     0.076 ,  1/10 ,    0.145])
    S = y[0]
    L = y[1]
    P = y[2]
    I = y[3]
    A = y[4]
    H = y[5]
    R = y[6]
    D = y[7]
    dSdt = -param[0]*(1-u)*(P+I+A*param[1])*S/param[2]
    dLdt=param[0]*(1-u)*(P+I+A*param[1])*S/param[2]-param[3]*L
    dPdt=param[3]*L-param[4]*P
    dIdt=param[4]*param[5]*P-param[6]*I
    dAdt=(1-param[5])*param[4]*P-param[7]*A
    dHdt=param[6]*param[8]*I-param[9]*H
    dRdt=param[6]*(1-param[8])*I+param[7]*A+(1-param[10])*param[9]*H
    dDdt=param[10]*param[9]*H
    return np.array([dSdt, dLdt,dPdt,dIdt,dAdt,dHdt,dRdt,dDdt])

def runge_kutta_4(y0, t0, t_end, dt,u):
    t = t0
    y = np.array(y0)
    t_values = [t]
    y_values = [y]
    while t < t_end:
        k1 = dydt(t, y,u)
        k2 = dydt(t + dt/2, y + dt/2 * k1,u)
        k3 = dydt(t + dt/2, y + dt/2 * k2,u)
        k4 = dydt(t + dt, y + dt * k3,u)
        
        y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dt
        
        t_values.append(t)
        y_values.append(y)
    
    return np.array(t_values), np.array(y_values)





t_values, y_values = runge_kutta_4(y0, t0, t_end, dt,u)

fig = plt.figure(figsize=(10,10))
plt.plot(t_values, y_values[:, 5],linestyle="",marker=".")
sol=realsystem()
t=sol.t
H=sol.y[5]
plt.plot(t,H)

plt.xlabel('t [Days]')
plt.ylabel('y [Number of patients in hospitals]')
plt.grid()
plt.show()
