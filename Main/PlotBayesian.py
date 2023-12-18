import matplotlib.pyplot as plt 
import csv 
  
x = [] 
y = [] 
  
index = 11

with open('./results/Bayes/Bayes58.csv','r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    iter = 0

    for row in lines: 
        if (iter > 20):
            x.append(iter) 
            y.append(round(float(row[index]), 4)) 
        iter += 1
  
plt.plot(x, y, color = 'silver', linestyle = "solid", 
         marker = '',label = "158") 

x = [] 
y = [] 

with open('./results/Bayes/Bayes25.csv','r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    iter = 0

    for row in lines: 
        if (iter > 20):
            x.append(iter) 
            y.append(round(float(row[index]), 4)) 
        iter += 1
  
plt.plot(x, y, color = 'gray', linestyle = "solid", 
         marker = '',label = "125") 

x = [] 
y = [] 

with open('./results/Bayes/Bayes38.csv','r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    iter = 0

    for row in lines: 
        if (iter > 20):
            x.append(iter) 
            y.append(round(float(row[index]), 4)) 
        iter += 1
  
plt.plot(x, y, color = 'darkgray', linestyle = "solid", 
         marker = '',label = "138") 

  
plt.xticks(rotation = 25) 
plt.xlabel('Iterations') 
plt.ylabel('AUROC-value') 
plt.ylim(0, 1)
plt.xlim(20, 120)
plt.title('Bayesian AUROC', fontsize = 20) 
plt.grid() 
plt.legend() 
plt.show() 