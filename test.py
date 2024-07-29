import pyomo.environ as pyo
import numpy as np

# Példa adatok
N = 5  # Időszakok száma
n = 2  # Intervenciók száma

# WTCmin és WTCmax vektorok definiálása
WTCmin = np.array([2, 3])
WTCmax = np.array([5, 7])

# Model létrehozása
model = pyo.ConcreteModel()

# Indexek
model.I = pyo.RangeSet(0, N-1)
model.J = pyo.RangeSet(0, n)

# Változók
model.v = pyo.Var(model.I, model.J, domain=pyo.Binary)
model.H = pyo.Var(model.I, model.J, domain=pyo.NonNegativeIntegers)

# Paraméterek
Bm = 1000  # Big-M konstans

# Célfüggvény (példa célfüggvény, amit minimalizálunk)
model.obj = pyo.Objective(expr=sum(model.v[i, j] for i in model.I for j in model.J), sense=pyo.minimize)

# Korlátozások
def big_m_constraint_1(model, j):
    return model.v[0, j] * (WTCmin[j] - model.H[0, j]) <= Bm * (model.v[0, j] * model.v[1, j])

model.big_m_con_1 = pyo.Constraint(model.J, rule=big_m_constraint_1)

def update_H(model, i, j):
    if i < N-1:
        return model.H[i+1, j] == (model.H[i, j] + model.v[i, j]) * model.v[i, j]
    else:
        return pyo.Constraint.Skip

model.update_H_con = pyo.Constraint(model.I, model.J, rule=update_H)

def max_wait_constraint(model, i, j):
    return model.H[i, j] <= WTCmax[j]

model.max_wait_con = pyo.Constraint(model.I, model.J, rule=max_wait_constraint)

# Big-M constraint for all time periods
def big_m_constraint_all(model, i, j):
    if i < N-1:
        return model.v[i, j] * (WTCmin[j] - model.H[i+1, j]) <= Bm * (model.v[i, j] * model.v[i+1, j])
    else:
        return pyo.Constraint.Skip

model.big_m_con_all = pyo.Constraint(model.I, model.J, rule=big_m_constraint_all)

# Megoldó futtatása
solver = pyo.SolverFactory('glpk')
solver.solve(model, tee=True)

# Eredmények kiírása
for i in model.I:
    for j in model.J:
        print(f'v[{i},{j}] = {pyo.value(model.v[i,j])}, H[{i},{j}] = {pyo.value(model.H[i,j])}')
