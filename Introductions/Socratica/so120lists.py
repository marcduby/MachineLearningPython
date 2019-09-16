
# test list
test_dict = [(i, i**2) for i in range(1, 10)]
print(test_dict)

# test new list with even numbers
test_dict2 = [(i, i**2) for i in range(1, 10) if i % 2 == 0]
print(test_dict2)

# using a lambda
lamb = lambda i : i**3 + 1
test_dict3 = [(i, lamb(i)) for i in range(1, 10) if i % 2 == 1]
print(test_dict3)

