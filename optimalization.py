import numpy as np
import pyomo.environ as pyo
import pyomo.opt as po
import matplotlib.pyplot as plt
import optimalization_modell as m
population=m.population
Latent=m.Latent
dt=m.dt
t_end=100.

x0 = [population-Latent, Latent,0.,0.,0.,0.,0.,0.]
x_init=np.zeros((8,int(t_end/dt)))
u_values={0: 0.,
          1: 0.1,
          2: 0.2,
          3: 0.3,
          4: 0.4,
          5: 0.5,
          6: 0.6,
          7: 0.7,
          8: 0.8,
          9: 0.9}



def create_model (x0param,xinitparam):
    model=pyo.ConcreteModel()
    model.horizont=range(int(t_end/dt))
    model.dim=range(8)
    
    model.u_values = pyo.Set(initialize=u_values.values())
    model.x=pyo.Var(model.horizont,model.dim,domain=pyo.NonNegativeReals)
    model.u=pyo.Var(model.horizont,domain=pyo.Reals)
    model.u_idx=pyo.Var(model.horizont,range(len(model.u_values)),domain=pyo.Binary)
    
    model.constraints = pyo.ConstraintList()
    model.u_idx_constraints = pyo.ConstraintList()
    model.u_rule_constraints=pyo.ConstraintList()
    model.hospital_capacity=pyo.ConstraintList()

    model.obj = pyo.Objective(rule=lambda model: obj_rule(model), sense=pyo.minimize)
        
    for i in model.horizont:
        for j in model.dim:
            model.x[i,j].value=xinitparam[j,i]
                       
    for j in model.dim:
            model.x[0,j].fix(x0param[j])


            

    #u_idx_const(model)
    #u_rule(model)
    #hospital_capacity_constraint(model)
    system_dynamic(model)
    return model


def hospital_capacity_constraint(model):
    for t in model.horizont:
        model.hospital_capacity.add(
            model.x[t,5]<=10000
        )


def obj_rule(model):
    return sum((model.u[t])**2 for t in model.horizont)

def real_model(model):
    #Ide a valós rendszer, folytonos jön
    return 0

def system_dynamic(model):
    t=0
    while t<t_end:
        y0=[None]*len(model.dim)
        for i in model.dim:
            y0[i]=model.x[t,i]
        res=m.runge_kutta_4_step(y0,model.u[t])
        for j in model.dim:
            if t<max(model.horizont):
                model.constraints.add(
                model.x[t+1,j]==res[j] 
                )
        t=t+dt        
    
def u_rule(model):
    for t in model.horizont:
        res=0
        for j in range(len(model.u_values)):
             res=model.u_idx[t,j]*u_values[j]+res
            
        model.u_rule_constraints.add(model.u[t]==res)
    


def u_idx_const(model):
    for t in model.horizont:
         model.u_idx_constraints.add(sum(model.u_idx[t,:])==1)




M=create_model(x0,x_init)
solution=pyo.SolverFactory('baron').solve(M, tee=True)

y_values=[None]*len(M.horizont)
for i in M.horizont:
    y_values[i]=M.x[i,0].value
    
t_values=np.linspace(0,len(M.horizont),len(M.horizont))

fig = plt.figure(figsize=(10,10))
plt.plot(t_values, y_values,linestyle="",marker=".")
    
plt.grid()    
plt.show()
print(y_values[99])
