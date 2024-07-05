def Runge_Kutta(fcn,t0, y0, t):
    h = 0.5
    n = (int)((t - t0)/h) 
    y = y0
    t = t0
    for i in range(1, n + 1):
        k1 = h * fcn(t, y)
        k2 = h * fcn(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * fcn(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * fcn(t + h, y + k3)
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return y
 