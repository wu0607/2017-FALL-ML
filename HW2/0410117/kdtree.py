import numpy
import math
import pprint
import sys


def main():
	pp = pprint.PrettyPrinter(indent=4)
	file = open(sys.argv[1],'r')
	data = []
	# loading train dataset
	for idx ,line in enumerate(file):
		line = line.strip("\r\n")
		data.append([float(element) for element in line.split(',')[2:-1]])
		data[idx].append(line.split(',')[-1])
		data[idx].append(line.split(',')[0])
	data = 	data[1:]
	
	file_test = open(sys.argv[2],'r')
	test_data = []
	# loading train dataset
	for idx ,line in enumerate(file_test):
		line = line.strip("\r\n")
		test_data.append([float(element) for element in line.split(',')[2:-1]])
		test_data[idx].append(line.split(',')[-1])
		test_data[idx].append(line.split(',')[0])
	test_data = test_data[1:]
	
	train_data = data
	#test_data = data[0:300]
	#print 'test_data',test_data,'\n'
	test_num = len(test_data)
	
	kdtree = build_kdtree(train_data, 0, 9)
	
	k_list = [1,5,10,100]
	
	for k in k_list:
		closest_knn = []
		for i in range ( 0,len(test_data) ):		
			closest_knn.append( get_knn(kdtree, test_data[i], k, 9, Euclidean_distance, True, 0, None) )
		
		clear_closest_knn = []	
		#print 'closest_knn:'
		#pp.pprint(closest_knn)
		
		for test in closest_knn:
			for element in test:
				clear_closest_knn.append( element[1] )
				
	
		#print 'clear_closest_knn: '
		#pp.pprint(clear_closest_knn)
		
		predi = vote(clear_closest_knn)
		#print 'predi: ', predi
		
		prediction = []
		for i in range (0,test_num):
			prediction.append( vote(clear_closest_knn[i*k : i*k+k]) )
			
		#print prediction
		correct = 0
		for i in range(0, test_num):
			if test_data[i][-2] == prediction[i]:
				correct += 1;
		print 'KNN accuracy:',float(correct)/test_num
		
		#print test_num,k
				
		for i in range(0,3):
			for j in range(0,k):
				print clear_closest_knn[i*k+j][-1],
			print
		print
	PCA(train_data, test_data)
 	
	
def build_kdtree(data, depth, dim):
	n = len(data)
	#dim = 10
	if n <= 0:
		return None
		
	axis = depth % dim	
	sorted_data = sorted(data, key=lambda data: data[axis])
	
	kdtree = [build_kdtree(sorted_data[:n/2], depth+1, dim), build_kdtree(sorted_data[n/2 +1:], depth+1, dim), sorted_data[n/2]]
		
	return kdtree
	
def Euclidean_distance(point1, point2):
	sum = 0
	for dim in range (0,len(point1)-2) :
		sum += (point1[dim]-point2[dim])**2
	
	return math.sqrt(sum)

def get_knn(kd_node, point, k, dim, Euclidean_distance, return_distances=True, i=0, heap=None):
    import heapq
    is_root = not heap
    if is_root:
        heap = []
    if kd_node:
        dist = Euclidean_distance(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        get_knn(kd_node[dx < 0], point, k, dim, Euclidean_distance, return_distances, i, heap)
        if dx * dx < -heap[0][0]: # -heap[0][0] is the largest distance in the heap
            get_knn(kd_node[dx >= 0], point, k, dim, Euclidean_distance, return_distances, i, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors if return_distances else [n[1] for n in neighbors]	
	
def vote(candidate):
	type=['cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'] 
	count = [0, 0, 0, 0, 0, 0, 0, 0]
	
	for i in range(0, len(candidate)):
		#print 'i: ',i,'  lenofcandidate:', len(candidate)
		#print candidate[i][-2]
		if candidate[i][-2] == 'cp':
			count[0] += 1
		elif candidate[i][-2] == 'im':
			count[1] += 1
		elif candidate[i][-2] == 'pp':
			count[2] += 1
		elif candidate[i][-2] == 'imU':
			count[3] += 1
		elif candidate[i][-2] == 'om':
			count[4] += 1
		elif candidate[i][-2] == 'omL':
			count[5] += 1
		elif candidate[i][-2] == 'imL':
			 count[6] += 1
		elif candidate[i][-2] == 'imS':
			count[7] += 1
	
	import operator
	index, value = max(enumerate(count), key=operator.itemgetter(1))
	#print index, value, count
	
	return type[index]

def PCA(train_data, test_data):
	train = []
	test = []
	test_num = len(test_data)
	for data in train_data:
		train.append( data[0:9] )
	train = numpy.array(train).astype(numpy.float)
	for data in test_data:
		test.append( data[0:9] )
	test = numpy.array(test).astype(numpy.float)
	
	#train = numpy.array(train_data).astype(numpy.float)
	mean = numpy.mean(train, axis=0) # axis=0: sum over column ; axis=1: sum over row
	mean_test = numpy.mean(test, axis=0)
	data_matrix = train.copy()
	test_matrix = test.copy()
	data_matrix = numpy.subtract(data_matrix, mean)
	test_matrix = numpy.subtract(test_matrix, mean_test)
	
	#Normalization??
	
	covariance_matrix = numpy.dot(data_matrix.T, data_matrix)
	
	eigen_values, eigen_vectors = numpy.linalg.eigh(covariance_matrix)
	#print eigen_values
	#print
	#print 'eigen_vectors: ',eigen_vectors
	
	projection = numpy.dot( data_matrix, numpy.array( \
		(eigen_vectors[:,8], eigen_vectors[:,7], eigen_vectors[:,6], eigen_vectors[:,5],\
		eigen_vectors[:,4], eigen_vectors[:,3], eigen_vectors[:,2]) ).T )
	testing = numpy.dot( test_matrix, numpy.array( \
		(eigen_vectors[:,8], eigen_vectors[:,7], eigen_vectors[:,6], eigen_vectors[:,5],\
		eigen_vectors[:,4], eigen_vectors[:,3], eigen_vectors[:,2]) ).T )
	'''	
	projection = numpy.dot( data_matrix, numpy.array( \
		(eigen_vectors[:,8], eigen_vectors[:,7], eigen_vectors[:,6], eigen_vectors[:,5],\
		eigen_vectors[:,4], eigen_vectors[:,3], eigen_vectors[:,2], eigen_vectors[:,1], eigen_vectors[:,0]) ).T )
	'''
	#print '\nprojection:', projection
	
	dim = len(projection[0])
	projection = projection.tolist()
	testing = testing.tolist()
	#print'projection:',len(projection),train_data[3][-2]
	for i in range(0,len(projection)):
		projection[i].append(train_data[i][-2])
		projection[i].append(train_data[i][-1])
	for i in range(0,len(testing)):
		testing[i].append(test_data[i][-2])
		testing[i].append(test_data[i][-1])
	pp = pprint.PrettyPrinter(indent=4)
	#pp.pprint(projection)
	kdtree_PCA = build_kdtree(projection, 0, dim)
	
	k = 5
	
	#print 'dim',dim
	#print testing[90:95]
	#print projection[90:95]
	closest_knn = []
	for i in range ( 0,len(testing) ):		
		closest_knn.append( get_knn(kdtree_PCA, testing[i], k, dim, Euclidean_distance, True, 0, None) )
	
	clear_closest_knn = []	
	#print 'closest_knn:'
	#pp.pprint(closest_knn)
	
	for test in closest_knn:
		for element in test:
			clear_closest_knn.append( element[1] )
			

	#print 'clear_closest_knn: '
	#pp.pprint(clear_closest_knn)
	
	prediction = []
	for i in range (0,test_num):
		prediction.append( vote(clear_closest_knn[i*k : i*k+k]) )
		
	#print prediction
	correct = 0
	for i in range(0, test_num):
		if testing[i][-2] == prediction[i]:
			correct += 1;
	#print correct
	
	'''
	for i in range(0,test_num):
		for j in range(0,k):
			print clear_closest_knn[i*k+j][-1],
		print
	print '''
	
	print 'K = 5, KNN_PCA accuracy: ',float(correct)/test_num
	
	
if __name__ == '__main__':
	main()
'''	
def closest_point(data, new_point):	
	best_point = None
	best_distance = None
	
	for current_point in data:
		current_distance = Euclidean_distance(new_point, current_point)
		
		if best_distance is None or current_distance < best_distance:
			best_distance = current_distance
			best_point = current_point
	
	return best_point'''