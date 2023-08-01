import time

d1 = {i: "i" for i in range(int(1e6))}
d2 = {i: "i" for i in range(int(34))}

t1 = time.time()
a = d1[20000]
print(time.time() - t1)
t1 = time.time()
a = d2[20]
print(time.time() - t1)