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


for i in range(20):
    with open('./results/Bayes/Bayes' + str(i) + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if(line_count > 21):
                BayesCombined.CSVWriteRowSeed(row, 100 + i)
            line_count += 1

for i in range(50):
    with open('./results/Bayes/Bayes' + str(i + 20) + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if(line_count > 20):
                BayesCombined.CSVWriteRowSeed(row, 100 + i + 20)
            line_count += 1

for i in range(30):
    with open('./results/Bayes/Bayes' + str(i + 70) + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if(line_count > 21):
                BayesCombined.CSVWriteRowSeed(row, 100 + i + 70)
            line_count += 1

BayesCombined.CSVClose()