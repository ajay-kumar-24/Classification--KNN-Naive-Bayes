import numpy as np
import pandas as pd
import sys,math
import csv,operator
import collections

mean = {}
sd = {}

def normalize(data,cols):
    mean = data.groupby(data[cols-1]).mean()
    var = data.groupby(data[cols-1]).var()
    count = data.groupby(data[cols-1]).size()
    prior = []
    for no in count:
        prior.append(float(no)/len(data.index))
    return [(mean.values),var.values,mean.index.values,prior]
def naive(data_t,mean,variance,classes,prior,cols,test):
    result = 0.0
    for index, rows in  data_t.iterrows():
        prod ={}
        for i in range(0,len(mean)):
            prob = 0.0
            prep = 0
            for j in range(1,cols-1):
                if (variance[i][j]) == 0.0:
                    if mean[i][j] == rows[j]:
                        prep = 0
                    else:
                        prep = float('-Inf')
                else:
                    prep = (-math.log(math.sqrt(2*math.pi*variance[i][j]))) - (math.pow((rows[j]-mean[i][j]),2)/(2*variance[i][j]))    
                prob += prep
            prod[classes[i]] = prob + math.log( (prior[i]))
        maxi = max(prod.iteritems(), key=operator.itemgetter(1))[0]
        if(int(maxi) == int(rows[cols-1])):
            result += 1
    print test+' accuracy  =   '+str(result/len(data_t.index))

def read_data(filename):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    data = list(lines)
	return data

def calculate_euclidean(X,Y,length):
	distance = 0.0
	for i in range(1,length):
		distance += pow((((float(X[i])- float(mean[i]))/float(sd[i]))-((float(Y[i])-float(mean[i]))/float(sd[i]))),2)
	return np.sqrt(distance)

def calculate_L1(X,Y,length):
	distance = 0.0
	for i in range(1,length):
		distance += abs((((float(X[i])- float(mean[i]))/float(sd[i]))-((float(Y[i])-float(mean[i]))/float(sd[i]))))
	return np.array(distance)

def find_class(classes):
	flag = 0
	for obj in sorted(classes, key=lambda k: len(classes[k]), reverse=True):
		if flag==0:
			le = len(classes[obj])
			mini = min(classes[obj])
			out = obj
			flag = 1
		else:
			if len(classes[obj])== le:
				if  mini < min(classes[obj]):
					continue
				else:
					out = obj
					mini = min(classes[obj])
			else:
				break
	return out

def train(X,t_data,k,L,types):
	train_data = list(t_data)
	new_list = [0]*len(t_data)
	for i in range(0,len(train_data)):
		if int(train_data[i][0])!= int(X[0]):
			if L==1:
				distance = calculate_L1(X,train_data[i],len(X)-1)
			else:
				distance = calculate_euclidean(X,train_data[i],len(X)-1)
			if distance==0 and types=='train':
				distance = float('Inf')
			new_list[i] = list(train_data[i])
			new_list[i].append(distance)
		else:
			new_list[i] = list(train_data[i])
			continue
	newA=sorted(new_list,key = operator.itemgetter(-1))[:k]
	classes = {}
	for i in newA:
		c = int(i[-2])
		if c not in classes.keys():
			classes[c] = [i[-1].tolist()]
		else:
			classes[c] += [i[-1].tolist()]
	out = find_class(classes)
	if int(out)!= int(X[-1]):
		return 0
	else:
		return 1

def normal(data):
	data = pd.DataFrame(data)
	for col in range(1,len(data.columns-1)):
		mean[col] = data[col].mean()
		sd[col] = data[col].std()

def training_accuracy(test_data,train_data,types):
	K = [1,3,5,7]
	L = [1,2]
	data_t = [train_data,test_data]
	for l in L:
		print 'L'+str(l)+' distance'
		for k in K:
			flag = 0
			sum = 0
			for i in range(0,len(test_data)):
				X = test_data[i]
				sum += train(X,train_data,k,l,types)
			accuracy = float(float(sum)/len(test_data))
			print 'Accuracy for K value -> '+str(k)+'        =      '+str(accuracy)
	
def main():
	train_data = read_data('train.txt')
	test_data = read_data('test.txt')
	data = pd.read_csv('train.txt', header=None)
	data_t = pd.read_csv('test.txt', header=None)
	d = [data,data_t]
	cols =  len(data.columns)
	stats  = normalize(data,cols)    
	mean = stats[0]
	variance = stats[1]
	classes = stats[2]
	prior = stats[3]
	print '\n###########----------Naive Bayes----------##########\n'
	naive(data,mean,variance,classes,prior,cols,'Training')
	naive(data_t,mean,variance,classes,prior,cols,'Testing')
	print '\n###########--------------KNN--------------##########\n'
	normal(data)
	print'@TRAINING ACCURACY\n'
	training_accuracy(train_data,train_data,'train')
	print'\n@TESTING ACCURACY\n'
	training_accuracy(test_data,train_data,'test')
	print
main()