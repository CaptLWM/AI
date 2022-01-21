import numpy as np

lst=[
    [1,2,3],
    [4,5,6],
    [7,8,9],
]

arr=np.array(lst)
print(arr)

print(arr[2,0:3])
print(arr [:,0])
print(arr[:,2:1])