import numpy as np
np.random.seed(13)
X=np.array([[1,0,1,0],[1,1,0,0]])
y=np.array([1,0,0,0])
y=y.reshape(1,4)  #better way? 
print(X,y,X.shape,y.shape)
W=np.random.randn(1,2)
B=np.random.randn(1,1)
print(W, W.shape, B, B.shape)
y_p_=W@X+B
print(y_p_, y_p_.shape)
y_p=1/(1+np.exp(-y_p_))
print(y_p, y_p.shape)

dW=-2*y_p*(y-y_p)*y_p*(1-y_p)@ X.T
dB=-2*y_p*(y-y_p)*y_p*(1-y_p) @ np.ones((1,4)).T
print(np.ones((1,4)).T, (np.ones((1,4)).T).shape)
print(dW, dW.shape)
print(dB, dB.shape)

def mseloss(W, B, X, y):
    y_p=W@X +B
    return np.sum((y-y_p)**2)

# print(mseloss(W,B,X,y))

lr=0.01
epochs=100
for epoch in range(epochs):
    print(epoch, mseloss(W, B, X,y)) 
    W+=lr*-2*y_p*(y-y_p)*y_p*(1-y_p)@ X.T
    B+=lr*-2*y_p*(y-y_p)*y_p*(1-y_p) @ np.ones((1,4)).T
