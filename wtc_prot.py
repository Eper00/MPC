import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
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
tmin=np.ones( len(u_values))*3
tmax=np.ones(len(u_values))*5

def model_create(Aparam,Bparam,xreqparam,x0param):
    model = pyo.ConcreteModel() 
    model.horizont=range(20)
    model.A=Aparam
    model.B=Bparam
    model.xreq=xreqparam
    
    model.u_values = pyo.Set(initialize=u_values.values())
    
    model.x=pyo.Var(model.horizont)
    model.u = pyo.Var(model.horizont,domain=model.u_values)
    model.u_idx=pyo.Var(model.horizont,range(len(model.u_values)),domain=pyo.Binary)
    model.H=pyo.Var(range(len(model.horizont)+1),range(len(model.u_values)),domain=pyo.NonNegativeIntegers)
    
    model.constraints = pyo.ConstraintList()
    model.u_idx_constraints = pyo.ConstraintList()
    model.u_rule_constraints=pyo.ConstraintList()
    model.wtc_constraints_downer=pyo.ConstraintList()
    model.wtc_constraints_upper=pyo.ConstraintList()
    model.counting_constraing=pyo.ConstraintList()
    

    model.obj = pyo.Objective(rule=lambda model: obj_rule(model), sense=pyo.minimize)
    
    model.x[0].fix(x0param)

    h_counting(model)
    u_idx_const(model)
    u_rule(model)
    system_dynamic(model)
    wtc_rule_downer(model)
    wtc_rule_upper(model)
    return model

def h_counting(model):
    for t in model.horizont:
        for j in  range(len(model.u_values)):
                model.counting_constraing.add(model.H[t+1,j]==(model.H[t,j]+model.u_idx[t,j])*model.u_idx[t,j])


def wtc_rule_upper(model):
    for t in model.horizont:
        for j in range(len(u_values)):
            model.wtc_constraints_upper.add(model.H[t,j]<=tmax[j])


def wtc_rule_downer(model):
  
    
    for t in model.horizont:
        
       
        if(t<max(model.horizont)):
            res1=0
            res2=0
            for j in range(len(u_values)):
                res1=res1+(model.u_idx[t, j] * (tmin[j] - model.H[t+1, j]))
                res2=res2+(model.u_idx[t, j] *model.u_idx[t+1, j]) 
        
        res2=res2*1200000
        model.wtc_constraints_downer.add(res1<=res2)

                  
                                    

def u_rule(model):
    for t in model.horizont:
        res=0
        for i in range(len(model.u_values)):
            res=model.u_idx[t,i]*u_values[i]+res
            
        model.u_rule_constraints.add(model.u[t]==res)
    


def u_idx_const(model):
    for t in model.horizont:
        model.u_idx_constraints.add(sum(model.u_idx[t,:])==1)

        


def obj_rule(model):
    return sum((model.x[t] - model.xreq)**2 for t in model.horizont)

def system_dynamic(model):
    for t in model.horizont:
        if t < max(model.horizont):
            model.constraints.add(model.x[t+1] == model.A * model.x[t] + model.B * model.u[t])
           

    

A = 1.1
B = 1.5




xreq=np.array([100.])
x0=np.array([0.])

total_time=20
total_time_u=np.linspace(0,total_time,total_time)
total_time_x=np.linspace(0,total_time+1,total_time)
xsystem = np.ones(20)
usystem=np.ones(20)

solver = pyo.SolverFactory('gurobi')




model=model_create(A,B,xreq,x0)

    
solution=solver.solve(model)
for i in range(total_time):
    usystem[i]=model.u[i].value
    xsystem[i] = model.x[i].value


model.H.display()
print('\n')
model.u_idx.display()
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.title("Beavatkozó jel")
plt.xlabel("Time [s]")
plt.ylabel("U")
plt.plot(total_time_u,np.squeeze(usystem),color="b",linestyle="",marker="o")
plt.grid()
y_ticks = np.arange(np.min(usystem), np.max(usystem) + 1, 1)
plt.yticks(y_ticks)
x_ticks = np.arange(0, len(total_time_u) + 1, 1)
plt.xticks(x_ticks)


plt.subplot(1,2,2)
plt.title("Állapotváltozó")
plt.xlabel("Time [s]")
plt.ylabel("X")
plt.plot(total_time_x,np.squeeze(xsystem),color="r",linestyle="-",marker="o")
plt.grid()
y_ticks = np.arange(np.min(xsystem), np.max(xsystem) + 1, 10)
plt.yticks(y_ticks)
x_ticks = np.arange(0, len(total_time_x) + 1, 1)
plt.xticks(x_ticks)
plt.show()
