import numpy as np
def dydt(S, L, P, I, A, H, u):
                    #0->beta   #1->delta    #2->N          #3->alpha    #4->p    #5->q     #6->ro_1   #7->ro_a  #8->eta  #9->h    #10->mikro
    param=np.array([1/3 ,     0.75 ,       9800000,    1/2.5 ,      1/3 ,    0.6 ,      1/4 ,     1/4 ,     0.076 ,  1/10 ,    0.145])
    dSdt = -param[0]*(1.-u)*(P+I+A*param[1])*S/param[2]
    dLdt=param[0]*(1.-u)*(P+I+A*param[1])*S/param[2]-param[3]*L
    dPdt=param[3]*L-param[4]*P
    dIdt=param[4]*param[5]*P-param[6]*I
    dAdt=(1-param[5])*param[4]*P-param[7]*A
    dHdt=param[6]*param[8]*I-param[9]*H
    dRdt=param[6]*(1-param[8])*I+param[7]*A+(1-param[10])*param[9]*H
    dDdt=param[10]*param[9]*H
    return [dSdt, dLdt,dPdt,dIdt,dAdt,dHdt,dRdt,dDdt]
def runge_kutta_4_step(model,t):

    dt=0.5
    
   
    k1 = dydt( model.x[t,0],model.x[t,1],model.x[t,2],model.x[t,3],model.x[t,4],model.x[t,5],model.u[t])
    k2 = dydt( model.x[t,0]+ dt/2 * k1[0],model.x[t,1]+ dt/2 * k1[1],model.x[t,2]+ dt/2 * k1[2],model.x[t,3]+ dt/2 * k1[3],model.x[t,4]+ dt/2 * k1[4],model.x[t,5]+ dt/2 * k1[5],model.u[t])
    k3 = dydt( model.x[t,0]+ dt/2 * k2[0],model.x[t,1]+ dt/2 * k2[1],model.x[t,2]+ dt/2 * k2[2],model.x[t,3]+ dt/2 * k2[3],model.x[t,4]+ dt/2 * k2[4],model.x[t,5]+ dt/2 * k2[5],model.u[t])
    k4 = dydt( model.x[t,0]+ dt * k3[0],model.x[t,1]+ dt * k3[1],model.x[t,2]+ dt * k3[2],model.x[t,3]+ dt * k3[3],model.x[t,4]+ dt * k3[4],model.x[t,5]+ dt * k3[5],model.u[t])
    
    
    res=[]
    for i in model.dim:
        res.append(model.x[t,i] + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]))
    return res

