import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import optimalization_modell as m
real_population=m.real_population
normal_population=m.normal_population
normal_latent=m.normal_latent
dt=m.dt
t_end=m.t_end
x0 = m.x0
x_init=np.zeros((6,int(t_end/dt)),dtype=float)
u_init=np.ones((int(t_end/dt)),dtype=float)*0.3
u_idx_init=np.ones((int(t_end/dt)),dtype=float)*30
# 30 napig semmi de tartsuk be mindent
# hetente változzon csak a beavatkozás
# rovarírtás



def create_model (x0param):
    model=pyo.ConcreteModel()
    model.horizont=range(int(t_end/dt))
    model.dim=range(6)
    model.u_kvantum=9
    
    
    model.x=pyo.Var(model.horizont,model.dim,domain=pyo.NonNegativeReals)
    model.u_idx=pyo.Var(model.horizont,domain=pyo.NonNegativeIntegers,bounds=(0,model.u_kvantum))
    model.u=pyo.Var(model.horizont,domain=pyo.NonNegativeReals)
    
    model.constraints = pyo.ConstraintList()
    model.u_rule_constraints=pyo.ConstraintList()
    model.hospital_capacity=pyo.ConstraintList()

    model.obj = pyo.Objective(rule=lambda model: obj_rule(model), sense=pyo.minimize)
        
    for i in model.horizont:
        model.u[i].value=u_init[i]
        model.u[i].value=u_idx_init[i]
        for j in range(1,len(model.dim)):
            model.x[i,j].value=x_init[j,i]
                       
    for j in model.dim:
            model.x[0,j].fix(x0param[j])
    
    u_rule(model)
    
    hospital_capacity_constraint(model)
    system_dynamic(model)
    return model


def hospital_capacity_constraint(model):
    for t in model.horizont:
        model.hospital_capacity.add(
            model.x[t,5]<=(10000/m.real_population)
        )


def obj_rule(model):
    return sum((model.u[t]**2) for t in model.horizont)


def system_dynamic(model):
    t=0.
    while t<t_end/dt:
        y0=[None]*len(model.dim)
        for i in model.dim:
            y0[i]=model.x[int(t/dt),i]
        res=m.runge_kutta_4_step(y0,model.u[int(t/dt)])
        for j in model.dim:
            if t<max(model.horizont):
                model.constraints.add(
                model.x[int(t+1/dt),j]==res[j] 
                )
        t=t+dt        
    
def u_rule(model):
    for t in model.horizont:
        model.u_rule_constraints.add(model.u[t] == 0.1*model.u_idx[t])
    

M=create_model(x0)
solution = pyo.SolverFactory('baron').solve(M, tee=True)
y_values=[None]*len(M.horizont)
u_values=[None]*len(M.horizont)
for i in M.horizont:
    y_values[i]=M.x[i,5].value*real_population
    u_values[i]=M.u[i].value
    
t_values=np.linspace(0,t_end-1,len(M.horizont))
hospital=m.real_model_simulation(u_values)



plt.figure(figsize=(12, 6))
plt.subplot(1,2,2)
plt.plot(t_values, hospital,color="b",linestyle="-",marker=".")

plt.subplot(1,2,2)
plt.plot(t_values,y_values,color="r",linestyle="",marker=".")
plt.grid()

plt.subplot(1,2,1)
plt.plot(t_values,u_values,color="b",linestyle="",marker="o")
plt.grid()

plt.show()
