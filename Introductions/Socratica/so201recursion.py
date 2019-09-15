
# code inspired from video: https://www.youtube.com/watch?v=Qk0zUZW-U_M


from functools import lru_cache

@lru_cache(maxsize = 1000)
def finonacci(n) :
    """recursive function to calculate the Fibonacci series"""

    # check inputs
    if (type(n) != int):
        raise TypeError("n must be a positive integer")
    if n < 1:
        raise ValueError("n must be a positive integer")


    # local variables
    value = 0

    # if first two, return 1
    if n == 0:
        value = 0
    elif n == 1:
        value = 1
    elif n == 2:
        value = 1
    elif n > 2:
        value = finonacci(n - 1) + finonacci(n - 2)

    # return
    return value



for n in range(1, 500):
    print(n, ": ", finonacci(n))

finonacci(-1)



