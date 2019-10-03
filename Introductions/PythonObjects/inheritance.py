
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
        """returns the string represenation of the developer class"""
        return '{}, {}'.format(super().__str__(), self.language)

class Manager(Employee):
    """class to encapsulate the manager data"""

    def __init__(self, first_name, last_name, salary, employees = None):
        super().__init__( first_name, last_name, salary)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def __str__(self):
        """returns the string represenation of the manager class"""
        return '{}, {}'.format(super().__str__(), self.print_employees())

    def print_employees(self):
        for employee in self.employees:
            print('---> {}'.format(employee))

    def add_employee(self, employee):
        if employee not in self.employees:
            self.employees.append(employee)

    def remove_employee(self, employee):
        if employee in self.employees:
            self.employees.remove(employee)

# maid program
dev1 = Developer('Marc', 'Dude', 70000, 'Python')
dev2 = Employee('Sally', 'Ride', 80000)

manager1 = Manager('Bill', 'Lumberg', 100000, [dev1, dev2])

# print the help on the classes
print(help(Developer))

# print the devs
print(dev1)
print(dev2)
print(manager1)
print()

dev3 = Developer('Bill', 'Nye', 90000, 'R')
manager1.add_employee(dev3)
manager1.remove_employee(dev2)

# print the devs
print(dev1)
print(dev2)
print(manager1)
print()

# check the instances
print('Is manager an employee: {}'.format(isinstance(manager1, Employee)))
print('Is manager a developer: {}'.format(isinstance(manager1, Developer)))
print('Is manager an manager: {}'.format(isinstance(manager1, Manager)))
print()

# check the subclassing
print('Is Manager a subclass of Employee: {}'.format(issubclass(Manager, Employee)))
print('Is Manager a subclass of Developer: {}'.format(issubclass(Manager, Developer)))
print('Is Employee a subclass of Developer: {}'.format(issubclass(Employee, Developer)))



