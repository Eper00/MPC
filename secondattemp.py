import numpy as np
import pyomo.environ as pyo
import pyomo.opt as po
import matplotlib.pyplot as plt

def real_system(x,u):
    return np.matmul(A,x)+B*u





def model_create(Aparam,Bparam,xreqparam,x0param):
    model = pyo.ConcreteModel() 
    model.horizont=range(5)
    model.dim=range(2)
    model.A=Aparam
    model.B=Bparam
    model.xreq=xreqparam
    
    model.x=pyo.Var(model.horizont,model.dim,domain=pyo.Reals)
    
    model.u_values = pyo.Set(initialize=u_values.values())
    
    model.u = pyo.Var(model.horizont,domain=model.u_values)
    model.u_idx=pyo.Var(model.horizont,range(len(model.u_values)),domain=pyo.Binary)
    
    model.constraints = pyo.ConstraintList()
    model.u_idx_constraints = pyo.ConstraintList()
    model.u_rule_constraints=pyo.ConstraintList()

    model.obj = pyo.Objective(rule=lambda model: obj_rule(model), sense=pyo.minimize)
    

    model.x[0,:].fix(x0param[0])
   
    u_idx_const(model)
    u_rule(model)
    system_dynamic(model)

    return model

def u_rule(model):
    for t in model.horizont:
        res=0
        for j in range(len(model.u_values)):
             res=model.u_idx[t,j]*u_values[j]+res
            
        model.u_rule_constraints.add(model.u[t]==res)
    


def u_idx_const(model):
    for t in model.horizont:
         model.u_idx_constraints.add(sum(model.u_idx[t,:])==1)

        

def obj_rule(model):
    return sum((model.x[t, d] - model.xreq[d])**2 for t in model.horizont for d in model.dim)

def system_dynamic(model):
    for t in model.horizont:
        if t < max(model.horizont):
            model.constraints.add(
                model.x[t+1, 0] == sum(model.A[0, j] * model.x[t, j] for j in model.dim) + model.B[0] * model.u[t]
            )
            model.constraints.add(
                model.x[t+1, 1] == sum(model.A[1, j] * model.x[t, j] for j in model.dim) + model.B[1] * model.u[t]
            )
    

A = np.array([[1,1.4],[1.4,1]])
B = np.array([1.5,1.5])

u_values={0:-10.,
          1:-8.,
          2:-6.,
          3:-4.,
          4:-2.,
          5:0.,
          6:2.,
          7:4.,
          8:6.,
          9:8.,
          10:10}


xreq=np.array([10.,10.])
x0=np.array([0.,0.])

total_time=20
total_time_u=np.linspace(0,total_time,total_time+1)
total_time_x=np.linspace(0,total_time+1,total_time+2)
xsystem = np.empty((0,2),float)
usystem=np.empty((0,1),float)

solver = po.SolverFactory('gurobi')
x_in=np.zeros((2,5),float)

for i in range(len(total_time_u)):
    if i ==0:
        model=model_create(A,B,xreq,x0)
        xsystem = np.vstack((xsystem,x0))
        x=x0
        
    results = solver.solve(model)
    
    
    
    usystem=np.vstack((usystem,np.array([model.u[0].value])))
    

    xreal=real_system(x,model.u[0].value)
    xsystem=np.vstack((xsystem,xreal))
    x=xreal
    x0=x
    x_in=np.array([[model.x[1,0].value,model.x[1,1].value],[model.x[2,0].value,model.x[2,1].value],[model.x[3,0].value,model.x[3,1].value],[model.x[4,0].value,model.x[4,1].value],[model.x[4,0].value,model.x[4,1].value]])    
    model=model_create(A,B,xreq,x0)


print(xsystem)
print(usystem)

