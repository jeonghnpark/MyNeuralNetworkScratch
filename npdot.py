import numpy as np
A=np.array([[1,2],[3,4], [5,6]])
print(A.shape)
B=np.array([7,8])
print(np.dot(A,B)) #B is broadcasted as np.array(([7,7,7][8,8,8]])

#개선된 softamx
a=np.array([1010, 1000, 990])
max_a=np.max(a)
print(max_a)
sum_exp=np.sum(np.exp(a-max_a))
softmax=np.exp(a-max_a)/sum_exp
print(softmax)

