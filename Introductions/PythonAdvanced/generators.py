
# script derived from Corey Schafer youtube channel example
# imports
import memory_profiler as mem_profile
import random
import time

# person class
class Person():
    def __init__(self, id, name, language):
        self.name = name
        self.language = language
        self.id = id

# define data
names = ['Marc', 'Bill', 'Sue', 'Joan', 'Jean']
languages = ['Python', 'Scala', 'Java', 'R', 'C']
number = 1000000

print(dir(mem_profile))

print('Memory before building: {}MB'.format(mem_profile.memory_usage()))

def create_people_list(num_people):
    result = []
    for i in range(1, num_people):
        result.append(Person(i, random.choice(names), random.choice(languages)))

    return result

def create_people_generator(num_people):
    for i in range(1, num_people):
        person = Person(i, random.choice(names), random.choice(languages))

        yield person


# build 100000 people list
t1 = time.clock()
people_list1 = create_people_list(number)
t2 = time.clock()

# log
print('Memory after building list: {}MB'.format(mem_profile.memory_usage()))
print('Time taken: {}s'.format(t2 - t1))

# build a generator
print()
print('Memory before building generator: {}MB'.format(mem_profile.memory_usage()))
t1 = time.clock()
people_generator = create_people_generator(number)
t2 = time.clock()

# log
print('Memory after building generator: {}MB'.format(mem_profile.memory_usage()))
print('Time taken: {}s'.format(t2 - t1))

# build a list from the generator
print()
print('Memory before building second list: {}MB'.format(mem_profile.memory_usage()))
t1 = time.clock()
people_list2 = list(people_generator)
t2 = time.clock()

# log
print('Memory after building second list: {}MB'.format(mem_profile.memory_usage()))
print('Time taken: {}s'.format(t2 - t1))
