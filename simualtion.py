import model as m
import runge_kutta as rk

t0=0

Model=m.Transition_model(m.population-m.Latent,m.Latent,m.parameters)
u=0 
