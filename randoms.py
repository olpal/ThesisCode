import random
import csv
import util as u

random_file="./randomsip.log"

random_tests=50
k_fold_range=1

random_variables=[]
ip=True

if ip ==True:
    for i in xrange(random_tests):
        line=[]
        # min sigma
        line.append(random.randint(1, 40))
        # max sigma
        line.append(random.randint(30, 100))
        # num sigma
        line.append(random.randint(1, 20))
        # overlap
        line.append(round(random.uniform(0.1, 0.99), 2))
        # log
        line.append(random.randint(0, 1) == True)
        # theshold
        line.append(round(random.uniform(0.1, 4.0), 2))
        # add the line
        random_variables.append(line)


else:
    for i in xrange(random_tests):
        line=[]
        #training batch size
        while True:
            value = random.randint(64,256)
            if value % 16 != 0:
                continue
            line.append(value)
            break
        #learning rate
        line.append("%f" % round(random.uniform(0.000001,0.1),6))
        #dropout rate
        line.append(round(random.uniform(0.4,0.99),2))
        #epochs
        line.append(random.randint(1500,2500))
        #convolutional layer count
        line.append(random.randint(3,7))
        #neuron multiplier
        line.append(random.randint(20,55))
        #convolutional filter
        line.append(random.randint(2,9))
        #random seed
        line.append(random.randint(0,10000))
        #add the line
        random_variables.append(line)


try:
    with open(random_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        if ip:
            for line in random_variables:
                line.append(0)
                writer.writerow(line)
                line.pop()
        else:
            for line in random_variables:
                i = 1
                for i in xrange(k_fold_range):
                    line.append(i+1)
                    line.append(0)
                    writer.writerow(line)
                    line.pop()
                    line.pop()
except:
    u.printf("Unable to write to variable file: {}".format(random_file))
