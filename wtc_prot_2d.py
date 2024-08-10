import numpy as np
import pyomo.environ as pyo
import pyomo.opt as po
import matplotlib.pyplot as plt

# Ezek azok a bevatkozási értékek amelyekkel beavatkozhatunk (U halmaz).
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
# Azok a vektorok melyek meghatározzák, hogy az egyes beavatkozások mennyi ideig lehetnek aktívak (W_{Tmin},W_{Tmax})
tmin=np.ones( len(u_values))*2
tmax=np.ones(len(u_values))*4

# Függvény amely az optimalizációs problémát hozza létre:
# Bemenet: A rendszert leíró dinamika, kiinduló állapot, illetve az inicializációs vektor
# Kimenet: Az optimaliálandó probléma (obcejt és a határtok)
def model_create(Aparam,Bparam,xreqparam,x0param,xinitparam):
    model = pyo.ConcreteModel() 
    model.horizont=range(30)
    model.dim=range(2)
    model.A=Aparam
    model.B=Bparam
    model.xreq=xreqparam
    
    model.x=pyo.Var(model.horizont,model.dim,domain=pyo.Reals)
    
    model.u_values = pyo.Set(initialize=u_values.values())
    
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
    
    for i in model.horizont:
        for j in model.dim:
            model.x[i,j].value=xinitparam[j,i]



    model.x[0,:].fix(x0param[0])
    h_counting(model)
    u_idx_const(model)
    u_rule(model)
    system_dynamic(model)
    wtc_rule_downer(model)
    wtc_rule_upper(model)
    return model
# Függvény amely a számoláló léptetésért felelős (9-es egyenlet)
def h_counting(model):
    for t in model.horizont:
        for j in  range(len(model.u_values)):
                model.counting_constraing.add(model.H[t+1,j]==(model.H[t,j]+model.u_idx[t,j])*model.u_idx[t,j])

# Függvény amely a wtc felső betartását hozza létre (11-es egyenlet)
def wtc_rule_upper(model):
    for t in model.horizont:
        for j in range(len(u_values)):
            model.wtc_constraints_upper.add(model.H[t,j]<=tmax[j])

# Függvény amely a wtc alső betartását hozza létre (10-es egyenlet)
def wtc_rule_downer(model):
  
    for t in model.horizont:
        
        if(t<max(model.horizont)):
            res1 = 0
            res2 = 0
            for j in range(len(u_values)):
                res1 = res1 + (model.u_idx[t, j] * (tmin[j] - model.H[t+1, j]))
                res2 = res2 + (model.u_idx[t, j] * model.u_idx[t+1, j]) 
        res2 = res2 * 1200000
        model.wtc_constraints_downer.add( res1 <= res2) 

# Függvény, amely biztosítja, hogy a bevatkozás csak a megadott U halmazból kerüljön ki (7-es egyenlet)
def u_rule(model):
    for t in model.horizont:
        res=0
        for j in range(len(model.u_values)):
             res=model.u_idx[t,j]*u_values[j]+res
            
        model.u_rule_constraints.add(model.u[t]==res)
    

# Függvény amely biztosístja hogy egy adott időpontban csak egy index legyen aktív (8-as egyenlet)
def u_idx_const(model):
    for t in model.horizont:
         model.u_idx_constraints.add(sum(model.u_idx[t,:])==1)

        
# Az optimalizlandó kifejezés (3-as egynelet
def obj_rule(model):
    return sum((model.x[t, d] - model.xreq[d])**2 for t in model.horizont for d in model.dim)
# A rendszer dinamikát megvalósító hatrátok (1-es egynelet)
def system_dynamic(model):
    for t in model.horizont:
        if t < max(model.horizont):
            model.constraints.add(
                model.x[t+1, 0] == sum(model.A[0, j] * model.x[t, j] for j in model.dim) + model.B[0] * model.u[t]
            )
            model.constraints.add(
                model.x[t+1, 1] == sum(model.A[1, j] * model.x[t, j] for j in model.dim) + model.B[1] * model.u[t]
            )
    
# A rendszer dinamikát megahatározó paraméterek
A = np.array([[1,1.4],[1.4,1]])
B = np.array([1.5,1.5])


# Az állapot amelybe a rendszert irányítani akarjuk
xreq=np.array([100.,100.])
# Kiniduló állapot 
x0=np.array([0.,0.])
xinit=np.zeros((2,30),float)

# Az  idő horizont hossza
total_time=30

total_time_u=np.linspace(0,total_time,total_time+1)
total_time_x=np.linspace(0,total_time+1,total_time+2)

# A solver típusának beállítása
solver = po.SolverFactory('gurobi')
# A konrét modell "példányosítása"
model=model_create(A,B,xreq,x0,xinit)
# A probléma megoldása 
results = solver.solve(model)
# Adatárolók melyekbe az eredményt kapjuk
xsystem = np.ones((len(model.dim),len(model.horizont)))
usystem=np.ones(len(model.horizont))
# A kapott adatok eltárolása
for i in model.horizont:
    for j in model.dim:
        xsystem[j,i]=model.x[i,j].value
    usystem[i]=model.u[i].value

# Vektorok melyek az adat kirajzoltatását segítik elő
xtime=np.linspace(0,len(model.horizont),len(model.horizont))
utime=np.linspace(0,np.shape(usystem)[0],np.shape(usystem)[0])
# És végül az adtok vizualizációja 
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(2, 1, 1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Control Signal")
plt.grid()
y_ticks = np.arange(np.min(usystem), np.max(usystem) + 1, 1)
plt.yticks(y_ticks)
x_ticks = np.arange(0, len(utime) + 1, 1)
plt.xticks(x_ticks)
ax.plot(utime,usystem,linestyle="",marker="o")
ax = fig.add_subplot(2, 1, 2,projection='3d')

ax.plot3D(xsystem[0,:], xsystem[1,:],xtime, 'green')
ax.set_zlabel("Time [s]")
ax.set_xlabel("x1 State")
ax.set_ylabel("x2 state")





plt.show()


