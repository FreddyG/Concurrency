import fileinput
from math import sqrt

def is_prime(n):
    if n % 2 == 0:
        return False

    for i in xrange(3, int(sqrt(n) + 1)):
        if n % i == 0:
            return False

    return True

for candidate in fileinput.input():
    if not is_prime( int(candidate) ):
        print candidate
