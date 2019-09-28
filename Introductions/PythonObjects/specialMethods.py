
# example derived from the Corey Schafer youtube channel -> highly recommended to learn python if already know java/OOP

class Employee:
    """A class to encapsulate employee information"""

    def __init__(self, first_name, last_name, current_pay):
        self.first_name = first_name
        self.last_name = last_name
        self.email = self.first_name + '.' + self.last_name + "fubar.com"
        self.current_pay = current_pay

    def __repr__(self):
        return "Employee('{}', '{}', {})".format(self.first_name, self.last_name, self.current_pay)

    def __str__(self):
        return 'name: {} with pay: {}'.format(self.fullname(), self.current_pay)

    def fullname(self):
        """returns the fullname of the employee"""
        return '{} {}'.format(self.first_name, self.last_name)

    def apply_raise(self, percentage):
        """applies a raise to the employee salary based on the input percentage"""
        self.current_pay = int(self.current_pay * (1 + percentage))

    def dummy(self):
        """dummy function to use the 'pass' keyword"""
        pass

employee1 = Employee('Marc', 'MacMaster', 50000)
employee2 = Employee('Susan', 'Doody', 60000)

print()
print("name: {} with pay: {}".format(employee1.fullname(), employee1.current_pay))
print("name: {} with pay: {}".format(employee2.fullname(), employee2.current_pay))
print()
print(employee1)
print(employee2)
print()
print(employee1.__repr__())
print(employee2.__repr__())


employee1.apply_raise(.12)

print()
print(employee1)
print(employee2)

# print employee class help
print()
print("printing class help")
print(dir(employee1))
print(help(employee1))

# print employee function help
print()
print("printing fullname() method help")
print("dir: {}".format(dir(employee1.fullname())))
print("help: {}".format(help(employee1.fullname())))




