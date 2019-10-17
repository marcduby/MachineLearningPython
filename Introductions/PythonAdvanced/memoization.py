
# this script is derived from the Socratica memoization example

# imports
# from functools import lru_cache

# fibonacci functions
# @Lru_cache
def slow_fibbobanni(input_n):
    """Method to calculate the Fibonnacci sequence given the input integer"""
    # check the inputs
    if type(input_n) != int:
        raise TypeError("The input must be an integer")
    if input_n < 1:
        raise ValueError("The input must be a positive integer")

    # local variables
    return_value = 0

    if input_n == 1:
        return_value = 1
    elif input_n == 2:
        return_value = 1
    elif input_n > 2:
        return_value = slow_fibbobanni(input_n - 1) + slow_fibbobanni(input_n - 2)

    return return_value

# set the value
limit = 20

# print out the result
for n in range(1, limit):
    print("Fibbonacci for plave {} is: {}".format(n, slow_fibbobanni(n)))

# try error inputs

# TODO - add in print, error catching

test = slow_fibbobanni(-1)
test = slow_fibbobanni("dude")
test = slow_fibbobanni(True)
