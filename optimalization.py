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
# többi állapotváltozás
# második hullám
# rovarírtás még
# terminális halmaz megkeresése
# ezután annek beépítése
x_init=np.ones((6,t_end),dtype=float)
k=100000

def create_model (x0param):
    model=pyo.ConcreteModel()
    model.horizont=range(t_end)
    model.dim=range(6)
    model.u_kvantum=10*k
    model.weeks=range(int((t_end-1)/7)+1)
        
    model.x=pyo.Var(model.horizont,model.dim,domain=pyo.NonNegativeReals)
    model.u=pyo.Var(model.weeks,domain=pyo.NonNegativeIntegers,bounds=(0,model.u_kvantum))
    
    model.constraints = pyo.ConstraintList()
    model.u_rule_constraints=pyo.ConstraintList()
    model.hospital_capacity=pyo.ConstraintList()

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
    for i in range(1,len(model.horizont)):
        for j in range(len(model.dim)):
            model.x[i,j].value=x_init[j,i]
                       
    for j in model.dim:
            model.x[0,j].fix(x0param[j])


    for i in range(1,4):
        if (len(model.weeks)>i):
            model.u[len(model.weeks)-i].fix(0)

   
        
        
        
    system_dynamic(model)


    
    return model


def hospital_capacity_constraint(model):
    for t in model.horizont:
        model.hospital_capacity.add(
            model.x[t,5]<=m.normal_max_patients
        )


def obj_rule(model):
    return sum((model.u[t]**2) for t in model.weeks)


def system_dynamic(model):
    x_temp=[None]*len(model.dim)
    for t in model.horizont:
        
        for i in model.dim:
            x_temp[i]=model.x[t,i]
        res=m.runge_kutta_4_step(x_temp,(0.1/k)*model.u[int(t/7)])
        for j in model.dim:

            if t < max(model.horizont):
                
                model.constraints.add(model.x[t+1,j]==res[j])
        if(t>t_end-20):
            model.hospital_capacity.add(model.x[t,5]<=m.end_max_patients)
        else:        
            model.hospital_capacity.add(model.x[t,5]<=m.real_max_patients)

        

    

M=create_model(x0)
solution = pyo.SolverFactory('baron')

solution=solution.solve(M, tee=True)
y_values=[np.float64]*len(M.horizont)
u_values=[np.float64]*len(M.horizont)
for i in M.horizont:
    y_values[i]=M.x[i,5].value
    u_values[i]=M.u[int(i/7)].value*(0.1/k)
    
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
M.weeks