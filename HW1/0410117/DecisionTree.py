import numpy
from math import log
from random import shuffle

class Node:
	def __init__(self,data):
		self.left = None
		self.right = None
		self.data = data
		self.threshold_indices = -1
		self.threshold = -1
		self.leaf = True
		self.pure = True
		self.label = -1
		#set label
		if len(data)>1:
			label = data[0][-1]
			for i in range(1, len(data)):
				if label != data[i][-1]:
					self.pure = False
		if len(data)>1 and self.pure:
			self.label = data[0][-1]
	def setThresholdIndices(self, index):
		self.threshold_indices = index
	def setThreshold(self, val):
		self.threshold = val
	def setLeft(self, data):
		self.leaf = False
		self.left = data
	def setRight(self, data):
		self.leaf = False
		self.right = data

#spilt data into a left/right branch
def split(data, threshold, feature_index):
	left = []
	right = []
	for datapoint in data:
		if datapoint[feature_index] <= threshold:
			left.append(datapoint)
		else:
			right.append(datapoint)
	return(left, right)
	
#calculate the entropy, the lower the better
def calc_entropy(data):
	count = [0, 0, 0]
	total = float(0)
	#print numpy.array(data).shape
	for datapoint in data:
		#print datapoint,'\n' 
		if datapoint[-1] == 'Iris-setosa':
			count[0] += 1
		elif datapoint[-1] == 'Iris-versicolor':
			count[1] += 1
		else: #Iris-virginica
			count[2] += 1
	total = count[0] + count[1] + count[2]

	entropy = float(0)
	for c in count:
		if c == 0:
			continue
		prob = float(c) / total
		entropy = entropy - prob * log(prob)
		
	return entropy
#choose the best feature from sepal/petal length/width
def calc_threshold(data):	
	best_feature_index = -1
	best_entropy = float('inf')
	best_thres = float('inf')
	
	for i in range(len(data[0][:-1])):
		(entropy, thres) = calc_lowest_entropy(data, i)
		if entropy < best_entropy:
			best_entropy = entropy
			best_feature_index = i
			best_thres = thres
	return(best_thres, best_feature_index)		
			
#specific feature_index's entropy
def calc_lowest_entropy(data, feature_index):	
	sort = sorted(data, key=lambda tup: tup[feature_index])#sort by row
	#print 'type(sort[0][0])',type(sort[0][0])
	best_entropy = float('inf')
	best_thres = float('inf')
	curr_entropy = float('inf')
	curr_thres = float('inf')
	
	for i in range(0,len(data)):
		if i < len(data)-1:
			curr_thres = ( (sort[i][feature_index])+(sort[i+1][feature_index]) )/2.0
			
		(left, right) = split(sort, curr_thres, feature_index)
		curr_entropy = calc_entropy(left)*float(len(left))/float(len(data))\
			+ calc_entropy(right)*float(len(right))/float(len(data))
	
		if curr_entropy < best_entropy:
			best_entropy = curr_entropy
			best_thres = curr_thres
	
	return(best_entropy, best_thres)

def find_impure_leaf(node):
	if node == None:
		return None
	if (node.pure == False) and (node.leaf==True):
		return node
	
	left_child = find_impure_leaf(node.left)
	if left_child != None:
		return left_child
	
	right_child = find_impure_leaf(node.right)	
	if right_child != None:
		return right_child
		
	return None
#ID3 Algorithm	
def ID3(root):
	#print 'ID3 start'
	curr_node = find_impure_leaf(root)
	while curr_node != None:
		(threshold, feature_index) = calc_threshold(curr_node.data)
		(left, right) = split(curr_node.data, threshold, feature_index)
		
		curr_node.setThreshold(threshold)
		curr_node.setThresholdIndices(feature_index)
		left_node = Node(left) 
		right_node = Node(right)
		curr_node.setLeft(left_node)
		curr_node.setRight(right_node)
		curr_node.leaf = False
		
		curr_node = find_impure_leaf(root)
	#print 'ID3 finish'		
	
		
def k_fold_validation(data):
	training_set = []
	testing_set = []
	
	data_1 = data[:int(len(data)*0.8)]
	training_set.append(data_1)
	data_2 = data[:int(len(data)*0.6)] + data[int(len(data)*0.8):]
	training_set.append(data_2)
	data_3 = data[:int(len(data)*0.4)] + data[int(len(data)*0.6):]
	training_set.append(data_3)
	data_4 = data[:int(len(data)*0.2)] + data[int(len(data)*0.4):]
	training_set.append(data_4)
	data_5 = data[int(len(data)*0.2):]
	training_set.append(data_5)
	
	data_1 = data[int(len(data)*0.8):]
	testing_set.append(data_1)
	data_2 = data[int(len(data)*0.6):int(len(data)*0.8)]
	testing_set.append(data_2)
	data_3 = data[int(len(data)*0.4):int(len(data)*0.6)]
	testing_set.append(data_3)
	data_4 = data[int(len(data)*0.2):int(len(data)*0.4)]
	testing_set.append(data_4)
	data_5 = data[:int(len(data)*0.2)]
	testing_set.append(data_5)
	
	return(training_set, testing_set)

def predict(datapoint, root):
	curr_node = root
	#print root.data
	#print list(curr_node.__dict__.keys())
	while not(curr_node.pure):
		threshold = curr_node.threshold
		feature_index = curr_node.threshold_indices
		if datapoint[feature_index] <= threshold:
			curr_node = curr_node.left
		else:
			curr_node = curr_node.right
	
	return curr_node.label
	
def calc_error(data, root):
	true_positive = [0.0, 0.0, 0.0]
	true_negative = [0.0, 0.0, 0.0]
	false_positive = [0.0, 0.0, 0.0]
	false_negative = [0.0, 0.0, 0.0]
	precision = [0.0, 0.0, 0.0]
	recall = [0.0, 0.0, 0.0]
	total_accuracy = [0.0, 0.0, 0.0]
	count_ground_truth = [0.0, 0.0, 0.0]
	errors = 0.0
	
	for datapoint in data:
		prediction = predict(datapoint, root)
		ground_truth = datapoint[-1]
		
		
		#if (datapoint[3]>1.8 and datapoint[0]>6 and prediction == 'Iris-versicolor'):
		#	prediction = 'Iris-virginica'
		
		if(ground_truth != prediction):
			errors += 1	
		if prediction == 'Iris-setosa':
			if ground_truth =='Iris-setosa':
				#print'1'
				true_positive[0] += 1
				true_negative[1] += 1
				true_negative[2] += 1
				count_ground_truth[0] += 1
			elif ground_truth == 'Iris-versicolor':
				#print'2'
				false_positive[0] += 1
				false_negative[1] += 1
				true_negative[2] += 1
				count_ground_truth[1] += 1
			elif ground_truth == 'Iris-virginica':
				#print'3'
				false_positive[0] += 1
				true_negative[1] += 1
				false_negative[2] += 1
				count_ground_truth[2] += 1
				
		elif prediction == 'Iris-versicolor':
			if ground_truth =='Iris-setosa':
				#print'4'
				false_negative[0] += 1
				false_positive[1] += 1
				true_negative[2] += 1
				count_ground_truth[0] += 1
			elif ground_truth == 'Iris-versicolor':
				#print'5'
				true_negative[0] += 1
				true_positive[1] += 1
				true_negative[2] += 1
				count_ground_truth[1] += 1
			elif ground_truth == 'Iris-virginica':
				#print'6'
				true_negative[0] += 1
				false_positive[1] += 1
				false_negative[2] += 1
				count_ground_truth[2] += 1
				
		elif prediction == 'Iris-virginica':
			if ground_truth =='Iris-setosa':
				#print'7'
				false_negative[0] += 1
				true_negative[1] += 1
				false_positive[2] += 1			
				count_ground_truth[0] += 1
			elif ground_truth == 'Iris-versicolor':
				#print'8'
				true_negative[0] += 1
				false_negative[1] += 1
				false_positive[2] += 1
				count_ground_truth[1] += 1
			elif ground_truth == 'Iris-virginica':
				#print'9'
				true_negative[0] += 1
				true_negative[1] += 1
				true_positive[2] += 1
				count_ground_truth[2] += 1
	accuracy = 1 - ( float(errors)/float(len(data)) )			
	for i in range(3):
		precision[i] = float(true_positive[i])/( float(true_positive[i])+float(false_positive[i]) )
		recall[i] = float(true_positive[i])/( float(true_positive[i])+float(false_negative[i]) )
		#total_accuracy[i] = ( float(true_positive[i])+float(true_negative[i]) ) \
		#				/( float(true_positive[i])+float(true_negative[i])+float(false_positive[i])+float(false_negative[i]) )
	#print float(len(data))
	#print 'true_positive',true_positive
	#print 'true_negative',true_negative
	#print 'false_positive', false_positive
	#print 'false_negative',false_negative
	#print accuracy, precision, recall
	
	#print ""
	return accuracy, precision, recall
	
def main():
	#def std_normalize(data):
	#	#print "haha"
	#	#print data[:, :-1]
	#	data = numpy.asarray(data)
	#	float_data = data[:, :-1].astype(numpy.float)
	#	avg = numpy.mean(float_data,0)
	#	std = numpy.std(float_data,0)
	#	data[:, :-1] = (float_data - avg)/std
	#	return data.tolist()
	
	
	file = open('iris.csv','r')
	data = []
	# loading iris dataset
	for idx ,line in enumerate(file):
		line = line.strip("\r\n")
		data.append([float(element) for element in line.split(',')[:-1]])
		data[idx].append(line.split(',')[-1])
	
	test_time = 1
	k_fold_num = 5
	total_precision = [0.0, 0.0, 0.0]
	total_recall = [0.0, 0.0, 0.0]
	total_acc = [0.0, 0.0, 0.0]
	acc_final = 0.0
	acc_precise = [0.0, 0.0, 0.0]
	acc_recall = [0.0, 0.0, 0.0]
	
	#std_normalize(data)
	
	
	shuffle(data)
	training_set, testing_set = k_fold_validation(data)
	
	for _ in range(test_time):
		for i in range(k_fold_num):	
			root = Node(training_set[i])
			#print training_set[i],i 
			ID3(root)
			#print Node(training_set[i])
			#print root.pure	
			
			accuracy, precision, recall = calc_error(testing_set[i], root)
			acc_final += accuracy
			for j in range(3):
				#print type(precision)
				acc_precise[j] += (float(precision[j]))
				#print 'precision[j]',precision[j]
				acc_recall[j] += (float(recall[j]))
				#print 'recall[j]',recall[j]
			
	
	print acc_final/(k_fold_num*test_time)
	print acc_precise[0]/(k_fold_num*test_time),'	',acc_recall[0]/(k_fold_num*test_time)
	print acc_precise[1]/(k_fold_num*test_time),'	',acc_recall[1]/(k_fold_num*test_time)
	print acc_precise[2]/(k_fold_num*test_time),'	',acc_recall[2]/(k_fold_num*test_time)
	
if __name__ == '__main__':
	main()
