import csv
import pandas
import numpy
from enum import Enum
import names
import matplotlib.pyplot as pyplot
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_colwidth', None)

#Set header
header = ['id', 'fullName', 'sex', 'born', 'enrolled', 'department', 'job', 'pay', 'completedProjects']

#Set enum values for Sex
class Sex(Enum):
    Male = 0
    Female = 1

#Set enum values for Department
class Department(Enum):
    Main = 0
    Auxiliary = 1
    Support = 2

#Set enum values for Job
class Job(Enum):
    Developer = 0
    Tester = 1
    Support = 2

#Write to file
with open('test.csv', 'w') as f:
    writer = csv.writer(f)
    #Write header
    writer.writerow(header)
    #Pick a random number of rows between 1001 and 2000
    nRows = numpy.random.randint(1001, 2000);
    #Write rows
    for i in range(0, nRows):
        row = []
        sex = numpy.random.choice(list(Sex)).name
        if sex == Sex.Male:
            row = [i, names.get_first_name('male') + " " + names.get_last_name(), sex, numpy.random.randint(1940, 2004), numpy.random.randint(2010, 2022), numpy.random.choice(list(Department)).name, numpy.random.choice(list(Job)).name, round(numpy.random.uniform(20000, 60000), 2), numpy.random.randint(0, 10)]
        else:
            row = [i, names.get_first_name('female') + " " + names.get_last_name(), sex, numpy.random.randint(1940, 2004), numpy.random.randint(2010, 2022), numpy.random.choice(list(Department)).name, numpy.random.choice(list(Job)).name, round(numpy.random.uniform(20000, 60000), 2), numpy.random.randint(0, 10)]
        writer.writerow(row)

pay = []
born = []
job = []
with open('test.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        #print(row)
        pay.append(float(row['pay']))
        born.append(int(row['born']))
        job.append(row['job'])

    print('\nPay (csv.DictReader):')
    print('Max: ', max(pay))
    print('Min: ', min(pay))
    print('Mean: ', numpy.mean(pay))
    print('Dispersion: ', numpy.ptp(pay))
    print('Standard deviation: ', numpy.std(pay))
    print('Median: ', numpy.median(pay))

    print('\nBorn (csv.DictReader):')
    print('Max: ', max(born))
    print('Min: ', min(born))
    print('Mean: ', numpy.mean(born))
    print('Dispersion: ', numpy.ptp(born))
    print('Standard deviation: ', numpy.std(born))
    print('Median: ', numpy.median(born))

    print('\nJob (csv.DictReader):')
    print('Mode: ', max(job))

dataframe = pandas.read_csv('test.csv')
print('\nDescription:\n', dataframe.describe())
print('\nCount:\n', dataframe.count())
#print('\nDuplicated:\n', dataframe.duplicated(keep = False))
#print('\nIs N/A:\n', dataframe.isna())

print('\nPay (Dataframe):')
print('Max: ', max(dataframe['pay']))
print('Min: ', min(dataframe['pay']))
print('Mean: ', numpy.mean(dataframe['pay']))
print('Dispersion: ', numpy.ptp(dataframe['pay']))
print('Standard deviation: ', numpy.std(dataframe['pay']))
print('Median: ', numpy.median(dataframe['pay']))

print('\nBorn (Dataframe):')
print('Max: ', max(dataframe['born']))
print('Min: ', min(dataframe['born']))
print('Mean: ', numpy.mean(dataframe['born']))
print('Dispersion: ', numpy.ptp(dataframe['born']))
print('Standard deviation: ', numpy.std(dataframe['born']))
print('Median: ', numpy.median(dataframe['born']))

print('\nJob (Dataframe):')
print('Mode: ', numpy.max(dataframe['job']))

pyplot.scatter(dataframe['enrolled'], dataframe['pay'])
pyplot.title('Dependence of salary on the year of enrollment')
pyplot.show()

pyplot.scatter(dataframe['completedProjects'], dataframe['pay'])
pyplot.title('Dependence of salary on the number of completed projects')
pyplot.show()

pyplot.scatter(dataframe['enrolled'], dataframe['completedProjects'])
pyplot.title('Dependence of the number of completed projects on the year of enrollment')
pyplot.show()
