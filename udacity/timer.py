import numpy as np

P = np.array([[0,0,1,1],[0,1,0,1]])
T = np.array([-1,1,1,-1])
Q = len(P)

n1 = 30
ep = 1

W1 = ep*(2*np.random.random((n1,2))-1)
b1 = ep*(2*np.random.random((n1,1))-1)
W2 = ep*(2*np.random.random((1,n1))-1)
b2 = ep*(2*np.random.random((1,1,))-1)

alfa = 0.001
epocas = 10000



for epoc in range(epocas):
    a2 = []
    sum = 0
    for q in range(Q):
        a1 = np.dot(W1,P.T[q])+b1
        a2.append(np.tanh(np.dot(W2,a1)+b2))
        e = T[q]-a2[q]
        print(e)