import numpy as np
import pyomo.environ as pyo
import pyomo.opt as po

def model_create(Aparam,Bparam,xreqparam,ulowerparam,uupperparam,x0param):
    model = pyo.ConcreteModel() 
    model.horizont=range(5)
    model.dim=range(1)
    model.A=Aparam
    model.B=Bparam
    model.xreq=xreqparam
    model.x=pyo.Var(model.horizont)
    model.u=pyo.Var(model.horizont,bounds=(ulowerparam,uupperparam),domain=pyo.Integers)
    model.constraints = pyo.ConstraintList()
    model.obj = pyo.Objective(rule=lambda model: obj_rule(model), sense=pyo.minimize)
    

    model.x[0].fix(x0param)
    

    system_dynamic(model)
    return model

def obj_rule(model):
    return sum((model.x[t] - model.xreq)**2 for t in model.horizont)

def system_dynamic(model):
    for t in model.horizont:
        if t < max(model.horizont):
            model.constraints.add(model.x[t+1] == model.A * model.x[t] + model.B * model.u[t])
           
            

A = 1.1
B = 1.5

u_values={
    0:-10,
    1:-8,
    2:-6,
    3:-4,
    4:-2,
    5:0,
    6:2,
    7:4,
    8:6,
    9:8,
    10:10,
}

xreq=np.array([110])
x0=np.array([0.])

uupper=10
ulower=-10
xsystem = np.empty((0,1),float)
usystem=np.empty((0,1),float)



model=model_create(A,B,xreq,ulower,uupper,x0)
xsystem = np.append(xsystem, np.array([model.x[0].value]), axis=0)

for i in range(100):

    solver = po.SolverFactory('gurobi')
    results = solver.solve(model)
   

    
    usystem = np.append(usystem,np.array([[model.u[0].value]]),axis=0)
    xsystem = np.append(xsystem, np.array([[model.x[1].value]]), axis=0)

    model=model_create(A,B,xreq,ulower,uupper,xsystem[-1])
    

print(xsystem)
print(usystem)