import numpy as np
import matplotlib.pyplot as plt

def dydt(t, y,param,u):
    S,L,P,I,A,H,R,D = y
    dSdt = -param[0]*(1-u)*(P+I+A*param[1])*S/param[2]
    dLdt=param[0]*(1-u)*(P+I+A*param[1])*S/param[2]-param[3]*L
    dPdt=param[3]*L-param[4]*P
    dIdt=param[4]*param[5]*P-param[6]*I
    dAdt=(1-param[5])*param[4]*P-param[7]*A
    dHdt=param[6]*param[8]*I-param[9]*H
    dRdt=param[6]*(1-param[8])*I+param[7]*A+(1-param[10])*param[9]*H
    dDdt=param[10]*param[9]*H
    return np.array([dSdt, dLdt,dPdt,dIdt,dAdt,dHdt,dRdt,dDdt])

def runge_kutta_4(dydt, y0, t0, t_end, dt,param,u):
    t = t0
    y = np.array(y0)
    step=(int)(t_end-t0)/dt
    t_values = [t]
    y_values = [y]
    while t < step:
        k1 = dydt(t, y,param,u)
        k2 = dydt(t + dt/2, y + dt/2 * k1,param,u)
        k3 = dydt(t + dt/2, y + dt/2 * k2,param,u)
        k4 = dydt(t + dt, y + dt * k3,param,u)
        
        y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dt
        
        t_values.append(t)
        y_values.append(y)
    
    return np.array(t_values), np.array(y_values)


population=9800000.
                    #0->beta   #1->delta    #2->N          #3->alpha    #4->p    #5->q     #6->ro_1   #7->ro_a  #8->eta  #9->h    #10->mikro
parameters=np.array([1/3 ,     0.75 ,       population,    1/2.5 ,      1/3 ,    0.6 ,      1/4 ,     1/4 ,     0.076 ,  1/10 ,    0.145])
Latent=1.
u=0

y0 = [population-Latent, Latent,0,0,0,0,0,0]  
t0 = 0.
t_end = 365.
dt = 0.5

t_values, y_values = runge_kutta_4(dydt, y0, t0, t_end, dt,parameters,u)


plt.plot(t_values*dt, y_values[:, 5], label='y1(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()
