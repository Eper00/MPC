import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



population=9800000.
Latent=10.
u=0.
y0 = np.array([population-Latent, Latent,0,0,0,0,0,0])  
t0 = 0.
t_end = 180.
t_span=np.array([t0,t_end])
dt = 1.
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
    return [dSdt, dLdt,dPdt,dIdt,dAdt,dHdt,dRdt,dDdt]

def scalar(scalar,array):
    res=[None]*len(array)
    for i in range(len(array)):
        res[i]=array[i]*scalar
    return res

def summation(first,second,third=None,fourth=None):
    if third is None:
        third = [0] * len(first)
    if fourth is None:
        fourth = [0] * len(first)
    S=[None]*len(first)
    for i in range(len(first)):
        S[i]=first[i]+second[i]+third[i]+fourth[i]

    return S

def runge_kutta_4(y0, t0, t_end, dt,u):
    t = t0
    y = y0
    t_values = [t]
    y_values = [y]
    while t<t_end:
        k1 = dydt(t, y, u)
        k2 = dydt(t + dt/2, summation(y , scalar(dt/2 , k1)),u)
        k3 = dydt(t + dt/2, summation(y , scalar(dt/2 , k2)),u)
        k4 = dydt(t + dt, summation(y , scalar(dt , k3)),u)
        K=summation(k1 , scalar(2,k2) , scalar(2 , k3) , k4)
        y = summation(y, scalar(dt/6,K))
        t = t + dt
        
        t_values.append(t)
        y_values.append(y)
    return t_values, y_values





t_values, y_values = runge_kutta_4(y0, t0, t_end, dt,u)

hospital=[]
for i in range(len(y_values)):
    hospital.append(y_values[i][0])

fig = plt.figure(figsize=(10,10))
plt.plot(t_values, hospital,linestyle="",marker=".")
sol=realsystem()
t=sol.t
H=sol.y[0]
plt.plot(t,H)
plt.legend(["Discretized system","Real system"])


plt.xlabel('t [Days]')
plt.ylabel('y [Number of patients in hospitals]')
plt.grid()
plt.show()
