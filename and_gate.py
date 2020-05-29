#class versio of train_and_gate.py
import numpy as np 
import matplotlib.pyplot as plt

class Logic_gate():
    def __init__(self):
        np.random.seed(13)
        self.W=np.random.randn(1,2)
        self.B=np.random.randn(1,1)
    def forward(self, x):
        return self.W@x+self.B
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def y_pred(self, X):
        y_p=(self.W)@X +self.B
        return int(y_p>0.5)

    def train(self, X,y):
        #X column-stacked input, 2d ndarray
        #y target column, 2d ndarray
        lr=0.5
        epochs=5000
        losses=[]
        xaxis=[]
        Ws=[]
        print('initial loss', (y-self.sigmoid(self.W@X +self.B))**2,((y-self.sigmoid(self.W@X +self.B))**2).mean())
        for epoch in range(epochs):
            y_=self.W@X +self.B
            y_p=self.sigmoid(y_)  #element by element 로 처리되나 확인할것. 
            loss=np.mean((y-y_p)**2)
            # loss=((y-y_p)**2).mean()
            losses.append(loss)
            xaxis.append(epoch)
            Ws.append(self.W)
            # print(epoch, loss, self.W[0,:])
            dW=-2*y_p*(y-y_p)*y_p*(1-y_p)@ X.T
            # dB=-2*y_p*(y-y_p)*y_p*(1-y_p) @ np.ones((1,4)).T
            dB=-2*y_p*(y-y_p)*y_p*(1-y_p) @ np.ones_like(y).T
            self.W+=-lr*dW
            self.B+=-lr*dB
            # print('loss', (y-self.sigmoid(self.W@X +self.B))**2,((y-self.sigmoid(self.W@X +self.B))**2).mean())
        plt.plot(xaxis, losses)
        plt.show()

    def train2(self, X,y):
        #X column-stacked input, 2d ndarray
        #y target column, 2d ndarray
        #update weight for each sample, 
        # cf. train1은 모든 sample에 대해서 dW, dB의 average를 사용

        lr=0.5
        epochs=1000
        losses=[]
        xaxis=[]
        Ws=[]
        print('initial y_p',self.sigmoid(self.W@X +self.B))
        print('initial loss', (y-self.sigmoid(self.W@X +self.B))**2)
        
        for epoch in range(epochs):
            
            for i,(x_1d,y_1d) in enumerate(zip(X.T,y.T)):
                x_1d_r=np.reshape(x_1d, (2,1)) #(2,1)-> (len(X),1)
                y_1d_c=np.reshape(y_1d, (1,1))
                # y_=self.W@x_1d_r +self.B
                y_p=self.sigmoid(self.W@x_1d_r +self.B)
                loss=(y-self.sigmoid(self.W@X +self.B))**2
            
                # Ws.append(self.W)
                # print(epoch, loss, self.W[0,:])
                dW=-2*y_p*(y_1d_c-y_p)*y_p*(1-y_p)@ x_1d_r.T  
                dB=-2*y_p*(y_1d_c-y_p)*y_p*(1-y_p) @ np.ones_like(y_1d_c).T
                self.W+=-lr*dW
                self.B+=-lr*dB
                if epoch %1==0:
                    # print(epoch, loss, self.W[0,:])
                    # print('loss', (y-self.sigmoid(self.W@X +self.B))**2,((y-self.sigmoid(self.W@X +self.B))**2).mean())
                    # print(epoch,'loss',(y-self.sigmoid(self.W@X +self.B))**2)
                    pass
            if epoch %100==0:
                pass
                # print(epoch, 'loss', (y-self.sigmoid(self.W@X +self.B))**2)
            loss=((y-y_p)**2).mean()
            # print(loss)
            losses.append(loss)
            xaxis.append(epoch)
        plt.plot(xaxis, losses)


#more readable way for input 
x1=np.array([1,1], dtype=float)
x2=np.array([1,0], dtype=float)
x3=np.array([0,1], dtype=float)
x4=np.array([0,0], dtype=float)
X=np.column_stack((x1,x2,x3,x4))
y=np.array([1,0,0,0], dtype=float) #AND gate

g=Logic_gate()

print('weight before train', g.W, g.B)
print('y  y_pred  y_target')
for x,y_ in zip(X.T,y):
    x=np.reshape(x, (2,1))
    print(g.forward(x),g.y_pred(x), y_)

g.train2(X, y)

print('weigh after train', g.W, g.B)

for x,y_ in zip(X.T,y):
    x=np.reshape(x, (2,1))
    print(g.forward(x),g.y_pred(x), y_)

plt.show()

