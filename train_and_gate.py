import numpy as np
import matplotlib.pyplot as plt

np.random.seed(13)
X=np.array([[1,0,1,0],[1,1,0,0]])

#more readable way for input 
x1=np.array([1,1])
x2=np.array([1,0])
x3=np.array([0,1])
x4=np.array([0,0])
X=np.column_stack((x1,x2,x3,x4))


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

# dW=-2*y_p*(y-y_p)*y_p*(1-y_p)@ X.T
# dB=-2*y_p*(y-y_p)*y_p*(1-y_p) @ np.ones((1,4)).T
# print(np.ones((1,4)).T, (np.ones((1,4)).T).shape)


def sigmoid(x):
    return 1/(1+np.exp(-x))

# def step(x):
#     if x > 0.5:
#         return 1
#     else:
#         return 0

def step(y):
    if y>0.5:
        return 1
    else:
        return 0


lr=0.1
epochs=50
losses=[]
xaxis=[]
Ws=[]
for epoch in range(epochs):
    y_=W@X +B
    y_p=sigmoid(y_)
    loss=np.mean((y-y_p)**2)
    # loss=((y-y_p)**2).mean()
    losses.append(loss)
    xaxis.append(epoch)
    Ws.append(W)
    print(epoch, loss, W[0][0], W[0][1])
    dW=-2*y_p*(y-y_p)*y_p*(1-y_p)@ X.T
    dB=-2*y_p*(y-y_p)*y_p*(1-y_p) @ np.ones((1,4)).T
    W+=-lr*dW
    B+=-lr*dB
    # print(acc(W,X,B,y))

y_p=W@X+B
# print(y_p.shape[1])
y_out=y_p.reshape((y_p.shape[1],))
print('yout', y_out)
for x,y in zip(X,y_out):
    print(x,y)
    

plt.plot(xaxis, losses)
plt.show()

