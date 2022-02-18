
# imports
import math

# input files

# methods
def calculate_abf(standard_error, effect_size, variance=0.396):
    ''' calculates the approximate bayes factor '''
    V = standard_error ** 2

    # calculate result
    left_side = math.sqrt(V / (V + variance))
    right_side = math.exp((variance * effect_size ** 2) / 2 * V * (V + variance))
    result = left_side * right_side

    # return
    return result

def convert_abf_to_probability(abf):
    ''' converts the approximate bayes factor to a probability '''
    PO = (0.05 / 0.95) * abf
    probability = PO / (1 + PO)

    # return
    return probability

# main
if __name__ == "__main__":
    pass
