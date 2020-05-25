import numpy as np
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)

suma=np.sum(a,axis=0)
suma.shape
print(suma)

sum_col_wise=np.sum(a,axis=1)
print(sum_col_wise)
print(sum_col_wise.shape)
t=sum_col_wise.reshape(3,1)
print(t.shape)

ones=np.ones((1,4))
print(ones, ones.shape)