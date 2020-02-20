
# import
import pandas as pd 

# set weights and bias
bias = -1.0
weight1 = 0.5
weight2 = 0.5

# inputs and outputs for AND perceptron
test_inputs = [(0,0), (0,1), (1,0), (1,1)]
correct_and_outputs = [False, False, False, True]
outputs = []

# function to test correctness of weights
def test_correctness(input, correct_out):
    for test_input, correct_output in zip(input, correct_out):
        linear_equation = weight1 * test_input[0] + weight2 * test_input[1] + bias
        output = int(linear_equation >= 0)
        is_correct_string = 'Yes' if output == correct_output else 'No'
        outputs.append([test_input[0], test_input[1], linear_equation, output, int(correct_output), is_correct_string])

    # print how done
    number_wrong = len([output_item[5] for output_item in outputs if output_item[5] == 'No'])
    output_df = pd.DataFrame(outputs, columns=['Input 1', 'Input 2', 'Linear Combination', 'Result', 'Expected', 'Is Correct'])
    if not number_wrong:
        print("Got all correct: {}".format(number_wrong))
    else:
        print("You got {} wrong".format(number_wrong))
    print(output_df.to_string(index = False))


# test the AND weights
print("testing AND perceptron")
test_correctness(test_inputs, correct_and_outputs)

# reset weights and bias
bias = 0.0
weight1 = 0.0
weight2 = -0.5

# test the NOT weights
correct_not_outputs = [True, False, True, False]
outputs = []
print("\ntesting NOT perceptron")
test_correctness(test_inputs, correct_not_outputs)
