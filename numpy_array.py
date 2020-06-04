# numpy array 연습
import numpy as np 
print(np.ones((3,2)))
#construct 2d arrary by stacking 1d array
x1=np.array([1,1])
x2=np.array([1,0])
x3=np.array([0,1])
x4=np.array([0,0])
X=np.column_stack((x1,x2,x3,x4))

#construct an array of ones with the same size of other array
a=[[1,2,3],[4,5,6]]
npa=np.array(a)
print(npa, type(npa))
ones_a=np.ones_like(npa)
print(ones_a)
zeros_a=np.zeros_like(npa)
print(zeros_a)

#construct a random array with the same size of other array
rand_a=np.empty_like(npa)
print(rand_a)

#loop columns in 2D array
#basically for loop for numpy array iterates row-wisely, so transpose it
#Since loop extracts each col of 2D array, it returns 1D array
for c in X.T:
    print(c)

#convert 1d ndarray to 2d array with same shape
x1_2d=np.reshape(x1, (2,1))
print(x1_2d, x1_2d.shape, x1_2d.ndim)

x1_2d=x1_2d.T
print(x1_2d, x1_2d.shape, x1_2d.ndim)


mat=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(mat)
for e in mat:
    for e1 in e:
        if e1>3:
            e1=1
        else:
            e1=0
print(mat)