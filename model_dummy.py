import numpy as np

class Transition_model:
    def __init__(self,S_init,L_init,parameters_param):
        self.S=S_init
        self.L=L_init
        self.P=10
        self.A=10
        self.I=10
        self.R=0
        self.H=0
        self.D=0
        self.param=parameters_param
    def dSdt(self,u):
        return -self.param[0]*(1-u)*(self.P+self.I+self.A*self.param[1])*self.S/self.param[2]
    def dLdt(self,u):
        return self.param[0]*(1-u)*(self.P+self.I+self.A*self.param[1])*self.S/self.param[2]-self.param[3]*self.L
    def dPdt(self):
        return self.param[3]*self.L-self.param[4]*self.P
    def dIdt(self):
        return self.param[4]*self.param[5]*self.P-self.param[6]*self.I
    def dAdt(self):
        return (1-self.param[5])*self.param[4]*self.P-self.param[7]*self.A
    def dHdt(self):
        return self.param[6]*self.param[8]*self.I-self.param[9]*self.H
    def dRdt(self):
        return self.param[6]*(1-self.param[8])*self.I+self.param[7]*self.A+(1-self.param[10])*self.param[9]*self.H
    def dDdt(self):
        return self.param[10]*self.param[9]*self.H

        
population=9800000
                    #0->beta   #1->delta    #2->N          #3->alpha    #4->p    #5->q     #6->ro_1   #7->ro_a  #8->eta  #9->h    #10->mikro
parameters=np.array([1/3 ,     0.75 ,       population,    1/2.5 ,      1/3 ,    0.6 ,      1/4 ,     1/4 ,     0.076 ,  1/10 ,    0.145])
Latent=10




   
    
    
    
    
    
    

