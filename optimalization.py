import numpy as np
import pyomo.environ as pyo
import pyomo.opt as po
import matplotlib.pyplot as plt
import optimalization_modell as m

def create_model (x0param,xinitparam):
    model=pyo.ConcreteModel()
    model.horizont=range(50)
    model.dim=range(8)
    
    model.u_values = pyo.Set(initialize=u_values.values())
    model.x=pyo.Var(model.horizont,model.dim,domain=pyo.Reals)
    model.u=pyo.Var(model.horizont,domain=pyo.Reals)
    model.u_idx=pyo.Var(model.horizont,range(len(model.u_values)),domain=pyo.Binary)
    
    model.constraints = pyo.ConstraintList()
    model.u_idx_constraints = pyo.ConstraintList()
    model.u_rule_constraints=pyo.ConstraintList()

    model.obj = pyo.Objective(rule=lambda model: obj_rule(model), sense=pyo.minimize)
        
    for i in model.horizont:
        for j in model.dim:
            model.x[i,j].value=xinitparam[j,i]
            model.x[0,j].fix(x0param[j])

    u_idx_const(model)
    u_rule(model)
    system_dynamic(model)
    return model


def obj_rule(model):
    return sum((model.u[t])**2 for t in model.horizont)

def real_model(model):
    return 0

def system_dynamic(model):

    for t in model.horizont:
          if t < max(model.horizont):
            res=m.runge_kutta_4_step(model,t)
            for j in model.dim:
                model.constraints.add(
                model.x[t+1,j]==res[j]
                    )            
    
def u_rule(model):
    for t in model.horizont:
        res=0
        for j in range(len(model.u_values)):
             res=model.u_idx[t,j]*u_values[j]+res
            
        model.u_rule_constraints.add(model.u[t]==res)
    


def u_idx_const(model):
    for t in model.horizont:
         model.u_idx_constraints.add(sum(model.u_idx[t,:])==1)

population=9800000
Latent=10

x0 = [population-Latent, Latent,0.,0.,0.,0.,0.,0.]
x_init=np.zeros((8,50))
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

M=create_model(x0,x_init)
M.x.display()