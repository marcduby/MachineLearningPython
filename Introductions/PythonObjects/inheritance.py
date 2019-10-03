
# examples derived from the Corey Schafer youtube channel; highly recommended

# employee class

class Employee:
    """class to encapsulate employee data"""

    def __init__(self, first_name, last_name, salary):
        self.first_name = first_name
        self.last_name = last_name
        self.email = self.first_name + "." + self.last_name + "@email.com"
        self.salary = salary

    def fullname(self):
        """returns the fullname"""
        return '{} {}'.format(self.first_name, self.last_name)

    def apply_raise(self, percentage):
        """applies a raise to the salary given the percentage given"""
        self.salary = int(self.salary * (1 + percentage))

    def __str__(self):
        """returns the string represenation of the employee class"""
        return '{}, {}, {}'.format(self.fullname(), self.email, self.salary)

class Developer(Employee):
    """class to encapsulate the developer data"""

    def __init__(self, first_name, last_name, salary, language):
        super().__init__( first_name, last_name, salary)
        self.language = language

    def __str__(self):
        """returns the string represenation of the employee class"""
        return '{}, {}'.format(super().__str__(), self.language)

# maid program
dev1 = Developer('Marc', 'Dude', 70000, 'Python')
dev2 = Employee('Sally', 'Ride', 80000)

# print the help on the classes
print(help(Developer))

# print the devs
print(dev1)
print(dev2)




