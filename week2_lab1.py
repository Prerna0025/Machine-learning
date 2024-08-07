import numpy as np
import time

a= np.zeros(4)
print(f"np.zeros(4): a={a}, a shape={a.shape}, a data type = {a.dtype}")
print('-----------------------------------------')
a= np.zeros((4,))
print(f"np.zeros(4): a={a}, a shape={a.shape}, a data type = {a.dtype}")
print('------------------------------------------')
a=np.random.random_sample(4)
print(f"np.random.random_sample(4): a={a}, a shape={a.shape}, a data type = {a.dtype}")
a=np.random.rand(4)
print(f"np.random.rand(4): a={a}, a shape={a.shape}, a data type = {a.dtype}")
a=np.array([5,4,2,3])
print(f"np.array([5,4,2,3]): a={a}, a shape={a.shape}, a data type = {a.dtype}")
a=np.array([5.,4,2,3])
print(f"np.array([5.,4,2,3]): a={a}, a shape={a.shape}, a data type = {a.dtype}")
#operation on vectors
a= np.arange(10)
print(f"np.arange(4): a={a}, a shape={a.shape}, a data type = {a.dtype}")
print(f"a[2]: {a[2]}")
print(f"a[-1]: {a[-1]}")

try:
    print(f"a[10]: {a[10]}")
except Exception as e:
    print(f"Error message: {e}")

print(f"a[2:7:1]: {a[2:7:1]}")
print(f"a[2:7:2]: {a[2:7:2]}")
print(f"a[:3]: {a[:3]}")
print(f"a[3:] : {a[3:]}")
print(f"a[:]: {a[:]}")
b = -a
print(f"b=-a: {b}")
print(f"np.sum(a): {np.sum(a)}")
print(f"np.mean(a): {np.mean(a)}")
print(f"a**2: {a**2}")

#vector vector operation
a= np.array([2, 4, 1, 3, 5])
b=np.array([-1, -1, 4, 5, 6 ])
c=np.array([3, 5])
print(f"a+b : {a+b}")
try:
    print(f"a+c: {a+c}")
except Exception as e:
    print(f"Error is: {e}")
    
#scalar vector operation
print(f"5*b: {5*b}")

#vector vector dot product

def my_dot(a,b):
    x=0
    for i in range(a.shape[0]):
        x=x+a[i]*b[i]
    return x
print(f"my_dot(a,b): {my_dot(a,b)}")
print(f"np.dot(a,b): {np.dot(a,b)}")

np.random.seed(1)
a=np.random.rand(100000) 
b=np.random.rand(100000)

tic = time.time()
c=np.dot(a,b)
toc = time.time()
print(f"np.dot(a,b): {c:.5f}")
print(f"vectorized duration: {1000*(toc-tic):.4f} ms")

tic = time.time()
c=my_dot(a,b)
toc=time.time()
print(f"my_dot(a,b): {c:.4f}")
print(f"loop version duration: {1000*(toc-tic): .4f} ms")
del(a)
del(b)

x = np.array([[1],[2],[3],[4]])
w=np.array([2])
c=np.dot(x[1],w)
print(x)
print(f"x[1] has shape {x[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

#matrix
a=np.zeros((1,5))
print(f"a shape = {a.shape} , a={a}")

a=np.zeros((2,1))
print(f"a shape = {a.shape}, a={a}")

a=np.random.random_sample((1,1))
print(f"a shape = {a.shape}, a={a}")

a=np.array([[5],[4],[3]])
print(f"a shape={a.shape}, np.array: a={a}")
a=np.array([[5],
            [4],
            [3]])
print(f"a shape = {a.shape} , np.array: = {a}")

#operation on matrices
a= np.arange(6).reshape(-1,2)
print(f"a.shape: {a.shape}, \na={a}")
print(f"\na [2,0].shape: {a[2,0].shape}, a[2,0] = {a[2,0]}")
a= np.arange(20).reshape(-1,10)
print(a)
print(f"a[0, 2:7:1] = {a[0, 2:7:1]}")
