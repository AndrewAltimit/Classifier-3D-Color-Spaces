import numpy as np
import cv2
import sys, os
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import time
from sklearn.metrics import confusion_matrix

### GLOBAL VARIABLES ###	
t = 4
b_w = 4
b_h = 4
cost = 5.0
		
		
# Given a directory name and an extension of files to search for,
# the function will return a sorted list of files in the folder.
def get_image_paths_from_folder(dir_name, extension):
	# Store current working directory, then change to desired directory
	cwd = os.getcwd()
	os.chdir(dir_name)
	
	# Get the image paths in the folder with the requested extension
	img_list = os.listdir('./')
	img_list = [dir_name + "/" + name for name in img_list if extension.lower() in name.lower() ] 
	img_list.sort()
	
	# Restore the working directory
	os.chdir(cwd)
	
	return img_list
		
		
# Given an image path, return the feature vector for this image (color histogram)
def process_image(image_path):
	# Read Image
	image = cv2.imread(image_path)
	
	# Determine Block Size
	M, N = image.shape[:2]
	block_width  = (2 * N) // (b_w + 1)
	block_height = (2 * M) // (b_h + 1)

	# Define window indices
	width_ind_start = (np.asarray(range(b_w)) * N) // (b_w + 1)
	width_ind_end = width_ind_start + block_width
	height_ind_start = (np.asarray(range(b_h)) * M) // (b_h + 1)
	height_ind_end = height_ind_start + block_height
	
	# Determine the histograms for each window and concatenate them together into histograms
	histograms = np.empty((0))
	for i in range(b_h):
		for j in range(b_w):
			block = image[height_ind_start[i]:height_ind_end[i], width_ind_start[j]:height_ind_end[j]]
			# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogramdd.html
			H, edges = np.histogramdd(block.reshape(-1, 3), bins=(t, t, t))
			histograms = np.concatenate((histograms, H.flatten()))

	return histograms
	
	
# Given a directory, return a list of sorted subdirectories
def get_subdirectories(directory):
	folders = []
	for i,j,y in os.walk(directory):
		if i == directory:
			continue
		folders.append(i)
	return sorted(folders)
	
	
# Given a dataset in dictionary format, reformat it to two lists, the classification list and the corresponding feature list
def format_dataset(dataset, msg):
	# Build Dataset
	X = []
	y = []
	for label in dataset:
		image_folder = dataset[label]
		for j in range(len(image_folder)):
			#print("{}... Label {} Image {}".format(msg, label, j))
			data = process_image(image_folder[j])
			X.append(data)
			y.append(label)
	return X, y

	
# Given a training dataset, build the model and return the classifier object
def train(training_dataset):
	# Format Training Dataset
	X, y = format_dataset(training_dataset, "TRAINING")
			
	# Linear SVC: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
	# Note: LinearSVC determines the multi-class strategy if y contains more than two classes. Defaults to one-vs-rest classifiers
	classifier = LinearSVC(C=cost)
	classifier.fit(X, y)
		
	return classifier
	
	
# Given a testing dataset and a classifier, classify each sample and determine the overall error rate
def test(classifier, test_dataset, classifications):
	X, y = format_dataset(test_dataset, "TESTING")
	

	scores = [0] * len(classifications)
	totals = [0] * len(classifications)
	label_pred_list = []
	print("CLASSIFICATION RESULTS: ")
	for i in range(len(X)):
		label = y[i]
		label_pred = classifier.predict(np.asarray(X[i]).reshape(1,-1))
		label_pred_list.append(label_pred)
		#print("pred: {} actual: [{}] MATCH: {}".format(label_pred, label, label_pred == label))
		if label_pred == label:
			scores[label] += 1
		totals[label] += 1
			
	print("Confusion Matrix:")
	print(confusion_matrix(y, label_pred_list))		
			
	print("\nClassification Performance:")
	for i in range(len(scores)):
		print("{:<15} -> Error Rate {:.2f}%".format(train_folders[i].rsplit("\\")[-1].upper() , 100 * (1 - (scores[i] / totals[i]))))
		
	print("Average Error Rate: {:.2f}%".format(100 * (1 - (sum(scores) / sum(totals)))))
	
	
# Given a list of folders and an extension of interest, return a dictionary of the image paths
# keys  -> classification label (number representing each unique folder)
# value -> list of image paths
def folder_list_to_image_dictionary(folders, extension):
	image_dictionary = dict()
	for i in range(len(folders)):
		image_dictionary[i] = get_image_paths_from_folder(folders[i], extension)
		
	return image_dictionary
	
	
if __name__ == "__main__":

	# Check if all proper input arguments exist
	if len(sys.argv) != 3:
		print("Improper number of input arguments")
		print("USAGE: main.py <train_dir> <test_dir>")
		sys.exit()
		
	start_time = time.time()
		
	# Get Subdirectories
	train_folders = get_subdirectories(sys.argv[1])
	test_folders = get_subdirectories(sys.argv[2])

	# Get dictionary of images: 
	training_dataset = folder_list_to_image_dictionary(train_folders, ".JPEG")
	test_dataset = folder_list_to_image_dictionary(test_folders, ".JPEG")

	# Build the model with the training data
	classifier = train(training_dataset)
	
	# Evaluate the model with the testing data
	test(classifier, test_dataset, train_folders)
	
	end_time = time.time()
	print("Total Runtime: {:.2f} Seconds".format(end_time - start_time))
	

	
