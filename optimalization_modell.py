import numpy as np
from scipy.integrate import solve_ivp
# Paraméterek a szimulációhoz
# A horizont melyen optimalizálunk
t_end=180
# Korrekció a numerikus stabilitásért (???)
correction=1000
real_population=9800000/correction
real_latent=50/correction
real_max_patients=20000/correction
# A mintavételezés idő tartalma
dt=1
# Kezdőállapot a rednszernek
x0 = [(real_population-real_latent)/real_population, real_latent/real_population,0.,0.,0.,0.,0.,0.]
# Ez a folytonos rendszert leíró függvény
# Bemenet : t,y paraméterei a differenciálegyenlet rendszernek, u a beavatkozó szekvencia
# Kiement : A rendszer válasza az elöző állapotokra, és beavatkozásra
def dydt(t, y,u):
   
                    #0->beta   #1->delta    #2->N          #3->alpha    #4->p    #5->q     #6->ro_1   #7->ro_a  #8->eta  #9->h    #10->mikro
    param=np.array([1/3 ,     0.75 ,       1,    1/2.5 ,      1/3 ,    0.6 ,      1/4 ,     1/4 ,     0.076 ,  1/10 ,    0.145])    
    S = y[0]
    L = y[1]
    P = y[2]
    I = y[3]
    A = y[4]
    H = y[5]
    R = y[6]
    D = y[7]
    dSdt = -param[0]*(1-u)*(P+I+A*param[1])*S/param[2]
    dLdt=param[0]*(1-u)*(P+I+A*param[1])*S/param[2]-param[3]*L
    dPdt=param[3]*L-param[4]*P
    dIdt=param[4]*param[5]*P-param[6]*I
    dAdt=(1-param[5])*param[4]*P-param[7]*A
    dHdt=param[6]*param[8]*I-param[9]*H
    dRdt=param[6]*(1-param[8])*I+param[7]*A+(1-param[10])*param[9]*H
    dDdt=param[10]*param[9]*H
    return [dSdt, dLdt,dPdt,dIdt,dAdt,dHdt,dRdt,dDdt]
# A folytonos rendszer megoldása egy időlépésre
# Bemenet: Egy elemei beavatkozás adott t-ben, t_span az adott dt idő alatt mennyi t-re számítsa ki a 
# megoldást a diefferenciálegyenlet solvere, y0 a kezdőállapot melyből indul a rendszer
# Kimenet: A differnciálegyenelt megoldása egy időlépés alatt 
def real_system_step(u,t_span,y0):
    times=np.linspace(t_span[0],t_span[1],1000)
    soln = solve_ivp(lambda t, y: dydt(t, y, u), t_span, y0, t_eval=times)
    return soln
# Függvény A folytonos rendszer megoldására, ezt lépteti a "real_system_step" függvényt
# Bemenet: A beavatkozóhel szekvenciája
# Kimenet: A kórházban lévő emberek számosságának alakulása az időben a folytonos modell alapján
def real_model_simulation(u_values):
    
    real_system=[]
    t0_step=0
    t_end_step=t0_step+dt
    t_span=np.array([t0_step,t_end_step])
    x=x0
    hospital=[]
    hospital.append(x[5]*real_population*correction)
        

    while(t0_step < t_end-1):
        sol=real_system_step(u_values[int(t0_step/dt)],t_span,x)
        real_system.append(sol)
        x=sol.y[:,-1]
        t0_step=t0_step+dt
        t_end_step=t_end_step+dt
        t_span=np.array([t0_step,t_end_step])
        hospital.append(x[5]*real_population*correction)    
    return hospital

# Segéd függvény, mivel pyomo könyvtárban az optimalizálandó változókat nem használhatunk numpy
# tömbbe, illetve nem hívhatunk meg rajtuk numpy függvényeket ezért implementáltam sajátokat
# ez a függvény egy tömböt szorozz meg egy skalárral
# Bemenet: A szorzandó tömb, illetve a skalár
# Kimenet: A skaálrral beszorzott tömb
def scalar(scalar,array):
    res=[None]*len(array)
    for i in range(len(array)):
        res[i]=array[i]*scalar
    return res
# Hasonoló okokból lett implementálva ahogyan a "scalar" függvény
# Bemenet: Az összeadanadó tömbök, a függvény elfogadja ,ha 2-3-vagy 4 tömböt akarunk összeadni egymással
# Kimenet: Az összeadott tömbök
def summation(first,second,third=None,fourth=None):
    if third is None:
        third = [0] * len(first)
    if fourth is None:
        fourth = [0] * len(first)
    S=[None]*len(first)
    for i in range(len(first)):
        S[i]=first[i]+second[i]+third[i]+fourth[i]
    return S
# Numerikusan megoldja a folytonos differenciálegyeneltet, ez a függvény lépteti a dinamikát, így kapjuk
# meg a x_k-->x_k+1-et, azaz felfoghatjuk, úgy, mint egy mintavétlező függvényt. (21-es egyenlet)
# Bemenet: A kezdő állapot adott k időben és a bevatkozás
# Kimenet: A k.-k időpontból a k+1-be lépünk 
def runge_kutta_4_step(y,u):
    t=0
    k1 = dydt(t, y, u)
    k2 = dydt(t + dt/2, summation(y , scalar(dt/2 , k1)),u)
    k3 = dydt(t + dt/2, summation(y , scalar(dt/2 , k2)),u)
    k4 = dydt(t + dt, summation(y , scalar(dt , k3)),u)
    K=summation(k1 , scalar(2,k2) , scalar(2 , k3) , k4)
    res = summation(y, scalar(dt/6,K))
    return res