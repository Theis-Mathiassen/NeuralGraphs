from Classes import CSVWriter 
import csv

BayesCombined = CSVWriter('BayesCombined', Seed=True)
BayesCombined.CSVOpen(Seed=True)

#with open('./results/Bayes0.csv', mode='r') as csv_file:
#    csv
"""with open('./results/Bayes/Bayes' + str(0) + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            print(row)"""


for i in range(55):
    with open('./results/Bayes/Bayes' + str(i) + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if(line_count > 1):
                BayesCombined.CSVWriteRowSeed(row, 100 + i)
            line_count += 1

for i in range(45):
    with open('./results/Bayes/Bayes' + str(i + 55) + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if(line_count > 0):
                BayesCombined.CSVWriteRowSeed(row, 100 + i + 55)
            line_count += 1
