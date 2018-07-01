import numpy as np

def Fx_lms(x, x_prima, y, error , w, c, mu, M):
    if len(x) >= M:
        x = x[-M:]
    else:
        x = np.concatenate((np.zeros(M-len(x)), x))
    x = x[::-1]
    y_actual = np.dot(x, w)
    x_prima_actual = np.sum((x[2])*c)
    y = np.concatenate((y,[y_actual]))
    x_prima = np.concatenate((x_prima, [x_prima_actual]))
    if len(x_prima) >= M:
        x_prima = x_prima[-M:]
    else:
        x_prima = np.concatenate((np.zeros(M - len(x_prima)), x_prima))
    x_prima = x_prima[::-1]

    if len(y) >= M:
        y = y[-M:]
    else:
        y = np.concatenate((np.zeros(M - len(y)), y))

    y = y[::-1]
    r = np.dot(c,y)
    e_prima = error - r
    w_nuevo = w - mu*error*x_prima
    c_nuevo = c + mu*e_prima*y

    return w_nuevo, c_nuevo, x_prima[::-1], y[::-1]





print(Fx_lms([140],[1],[1],40,np.zeros(256),np.zeros(256),0.35,256))