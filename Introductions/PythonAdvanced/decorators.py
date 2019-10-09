
# derived off of example in Corey Schafer youtube channel

# create decorator function
def decorator_function(inner_function):
    def wrapper_function():
        print("Running decorator1 before '{}'".format(inner_function.__name__))
        return inner_function()
    return wrapper_function

def display_function():
    print('running display function')

@decorator_function
def display2_function():
    print('running display2 function')

# run the functions
display = decorator_function(display_function)
display()

# run decorated function
print()
display2 = display2_function
display2()
