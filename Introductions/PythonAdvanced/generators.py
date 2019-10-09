
# imports
import mem_profile
import random
import time

# person class
class Person():
    def __init__(self, id, name, language):
        self.name = name
        self.language = language
        self.id - id

# define data
names = ['Marc', 'Bill', 'Sue', 'Joan', 'Jean']
languages = ['Python', 'Scala', 'Java', 'R', 'C']

print('Memory before building: {}MB'.format(mem_profile.memory_usage_resource()))

def people_list(num_people):
    result = []
    for i in range(1, num_people):
        result.append(Person(i, random.choice(names), random.choice(languages)))

    return result


# build 100000 people list
number = 100000
t1 = time.clock()
people_list = people_list(number)
t2 = time.clock()

# log
print('Memory after building: {}MB'.format(mem_profile.memory_usage_resource()))
print('Time taken: {}s'.format(t2 - t1))

