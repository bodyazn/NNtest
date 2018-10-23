import numpy as np
l = np.array([1, 2, 3])
l = l * 3
print(l)

class crazy_int:

    def __init__(self, x):
        self.value = x

    def __mul__(self, other):
        return (other.value * -self.value * 10)


class speclist(list):
    def __mul__(self, other):

        for i in range(0, len(self)):
            self[i] *= other

ci1 = crazy_int(2)
ci2 = crazy_int(8)

sl = speclist()
sl.append(1)
sl.append(2)

sl2 = sl * 100
print(sl, sl2)

print (ci1 * ci2)

def fib(num):
    fb2 = 1
    fb1 = 1
    for i in range(2, num):
        cur = fb1 + fb2
        fb2 = fb1
        fb1 = cur
    return fb1 * (num != 0)

#print(fib(100))

def emty_return():
    return

print(emty_return())