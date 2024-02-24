import time
import random
import string

d1 = {i: i for i in range(int(1e6))}
d2 = {i: i for i in range(int(34))}

d3 = {''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10000)): i for i in range(int(2000))}
d4 = {i: i for i in range(int(2000))}

t1 = time.time()
a = d1[20000]
print(time.time() - t1)

t2 = time.time()
a = d2[20]
print(time.time() - t2)

key = list(d3.keys())[100]

t3 = time.time()

a = d3[key]
print(time.time() - t3)
t4 = time.time()
a = d4[20]
print(time.time() - t4)