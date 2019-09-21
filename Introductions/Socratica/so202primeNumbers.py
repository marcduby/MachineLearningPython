import time

def is_prime_v1(n):
    """function to indicate if a number is prime"""
    if n == 1:
        return False

    for divisor in range(2, n-1):
        if n % divisor == 0:
            return False

    return True

# get start time
time0 = time.time()
for n in range(1, 30):
    print(n, is_prime_v1(n))
time1 = time.time()
print("time required v1: ", time1 - time0)



# devfest/Lehman0921



