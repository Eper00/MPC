import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



t0 = 0.
t_end = 50.
t_span=np.array([t0,t_end])
times=np.linspace(t_span[0],t_span[1],1000)
t=np.linspace(t0,t_end,1000)
def realsystem(y0_param):
    soln=solve_ivp(dydt,t_span,y0_param,t_eval=times)
    return soln
def dydt(t, y,u=0):
   
                    #0->beta   #1->delta    #2->N          #3->alpha    #4->p    #5->q     #6->ro_1   #7->ro_a  #8->eta  #9->h    #10->mikro
    param=np.array([1/3 ,     0.75 ,       9800000,    1/2.5 ,      1/3 ,    0.6 ,      1/4 ,     1/4 ,     0.076 ,  1/10 ,    0.145])
    S = y[0]
    L = y[1]
    P = y[2]
    I = y[3]
    A = y[4]
    H = y[5]
    #R = y[6]
    #D = y[7]
    dSdt = -param[0]*(1-u)*(P+I+A*param[1])*S/param[2]
    dLdt=param[0]*(1-u)*(P+I+A*param[1])*S/param[2]-param[3]*L
    dPdt=param[3]*L-param[4]*P
    dIdt=param[4]*param[5]*P-param[6]*I
    dAdt=(1-param[5])*param[4]*P-param[7]*A
    dHdt=param[6]*param[8]*I-param[9]*H
    #dRdt=param[6]*(1-param[8])*I+param[7]*A+(1-param[10])*param[9]*H
    #dDdt=param[10]*param[9]*H
    return np.array([dSdt, dLdt,dPdt,dIdt,dAdt,dHdt])