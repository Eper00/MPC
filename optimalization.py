import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import optimalization_modell as m
import mapping as mp
dt=m.dt
#Antal Péter git repo
terminal_state=mp.get_terminal_state()
# Paraméterek a szabályozásnak:
# Az idő horizont melyen irányítani szeretnénk 
t_end=m.t_end
# Kezdőállapot felvétele
x0 = m.x0
# inicializációs az optimalizációnak
x_init=np.ones((8,t_end),dtype=float)
# Kvantálása a beavatkozó jelnek
k=1000
# Ez a függvény felelős az optimalizációs probléma felépítésért
# Bemenet: Kezdőállapot a dinamikának
# Kimenet: Az optimalizálandó modell
def create_model (x0param):
    model=pyo.ConcreteModel()
    model.horizont=range(t_end)
    model.dim=range(8)
    model.u_kvantum=10*k
    model.weeks=range(int((t_end-1)/7)+1)
        
    model.x=pyo.Var(model.horizont,model.dim,domain=pyo.NonNegativeReals)
    model.u=pyo.Var(model.weeks,domain=pyo.NonNegativeIntegers,bounds=(0,model.u_kvantum))
    
    model.constraints = pyo.ConstraintList()
    model.hospital_capacity=pyo.ConstraintList()


    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
    for i in range(1,len(model.horizont)):
        for j in range(len(model.dim)):
            model.x[i,j].value=x_init[j,i]
                       
    for j in model.dim:
            model.x[0,j].fix(x0param[j])    
 
    system_dynamic(model)
    return model

# Az optimalizálandó kifejezés, minimalizálni akarjuk az eltérést a terminális állapottól, 
# illetve a beavatkozásokat is minimalizálni akarjuk (20-as egyenlet)
def obj_rule(model):
    
    return sum(((model.u[int(t/7)]*(0.1/k)**2+(model.x[t,0]-terminal_state[0])**2)) for t in model.horizont)

# A dinamika betartásáért felelős függvény (21-es egyenlet)
def system_dynamic(model):
    x_temp=[None]*len(model.dim)
    for t in model.horizont:
        # Korlátozzuk a kórházak kapacitásást is (22-es egyenlet)
        model.hospital_capacity.add(model.x[t,5]<=m.real_max_patients/m.real_population)
        for i in model.dim:
            x_temp[i]=model.x[t,i]
        # A diferenciálegyenlet rendszer numerikus megoldásának léptetésért felelős 
        res=m.runge_kutta_4_step(x_temp,(0.1/k)*model.u[int(t/7)])
        for j in model.dim:

            if t < max(model.horizont):
                
                model.constraints.add(model.x[t+1,j]==res[j])
        
    
    
    
# Létrehozzuk a problémát
M=create_model(x0)
# Kiválasztjuk a solvert
solution = pyo.SolverFactory('baron')
# És megoldjuk a solver segítségével 
solution=solution.solve(M, tee=True)
# Tömbök az adatok eltárolásához
s_values=[np.float64]*len(M.horizont)
l_values=[np.float64]*len(M.horizont)
p_values=[np.float64]*len(M.horizont)
i_values=[np.float64]*len(M.horizont)
a_values=[np.float64]*len(M.horizont)
h_values=[np.float64]*len(M.horizont)
r_values=[np.float64]*len(M.horizont)
d_values=[np.float64]*len(M.horizont)
u_values=[np.float64]*len(M.horizont)
# Az adatok eltárolása, illetve a numerikus adatok korrekciója a normálás aés korrigálása érdekében
for i in M.horizont:
    s_values[i]=M.x[i,0].value*m.real_population*m.correction
    l_values[i]=M.x[i,1].value*m.real_population*m.correction
    p_values[i]=M.x[i,2].value*m.real_population*m.correction
    i_values[i]=M.x[i,3].value*m.real_population*m.correction
    a_values[i]=M.x[i,4].value*m.real_population*m.correction
    h_values[i]=M.x[i,5].value*m.real_population*m.correction
    r_values[i]=M.x[i,6].value*m.real_population*m.correction
    d_values[i]=M.x[i,7].value*m.real_population*m.correction

    u_values[i]=M.u[int(i/7)].value*(0.1/k)
    
# Az adatok vizualizáció érdekében létrehozott tömb
t_values=np.linspace(0,t_end-1,len(M.horizont))
# A kórházban lévő szmosságot a folytonos modellen is szimuláljuk, így ellenörizzük, hogy valóban jók a számításaink
hospital=m.real_model_simulation(u_values)

# Végül az adtok vizualizációja
plt.figure(figsize=(12, 12))
plt.subplot(2,2,3)
plt.plot(t_values, hospital,color="k",linestyle="-",marker=".")
plt.plot(t_values,h_values,color="m",linestyle="",marker=".")
plt.legend(['The real system respond ','The predicted respond' ])
plt.xlabel("Time [days]")
plt.ylabel("Cardinality of the set [sample]")
plt.grid()
plt.subplot(2,2,2)
plt.plot(t_values,l_values,color="b",linestyle="-",marker=".")
plt.plot(t_values,p_values,color="g",linestyle="-",marker=".")
plt.plot(t_values,i_values,color="r",linestyle="-",marker=".")
plt.plot(t_values,a_values,color="c",linestyle="-",marker=".")
plt.plot(t_values,h_values,color="m",linestyle="-",marker=".")
plt.plot(t_values,d_values,color="k",linestyle="-",marker=".")
plt.legend(['Latent','Pre-symptomatic ','Symptomatic infected','Symptomatic infected but will recover','Hospital','Died'])
plt.xlabel("Time [days]")
plt.ylabel("Cardinality of the set [sample]")
plt.grid()
plt.subplot(2,2,4)
plt.plot(t_values,s_values,color="b",linestyle="-",marker=".")
plt.plot(t_values,r_values,color="g",linestyle="-",marker=".")
plt.legend(['Susceptibles','Recover'])
plt.xlabel("Time [days]")
plt.ylabel("Cardinality of the set [sample]")
plt.grid()

plt.subplot(2,2,1)
plt.plot(t_values,u_values,color="b",linestyle="",marker="o")
plt.legend(['Control signal'])
plt.xlabel("Time [days]")
plt.ylabel("Control scenarios")
plt.grid()

plt.show()
