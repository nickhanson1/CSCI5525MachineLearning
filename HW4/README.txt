Nick Hanson
5458741
hans6064@umn.edu

adaboost.py
USE: adaboost.py <filename>
	Will run the adaboost algoritm for 100 iterations on the dataset give. Will remove the first column of features,
	in order to remove patient IDs from the cancer dataset. Will output the training and test error rates for each 
	iteration.

rf.py
USE: rf.py <filename>
	Will run the random forest algoritm for 100 iterations on the dataset give. Will remove the first column of 
	features, in order to remove patient IDs from the cancer dataset. Will output the training and test error rates
	for each iteration of the algortihm when m=3, then will run the algorithm for 100 iterations on each of 
	m=2,3,4,5,6,7,8,9 and output the final training and test error for each.

kmeans.py
USE: kmeans.py <filename>
	Will run the kmeans algorithm for 15 iterations on the provided PNG file. Will segment the image with k=3,5,7.
	Will output the value of the distortion measure to the terminal for each iteration, and will output the generated
	segmented images to the folder /segmented_images/ in the root directory.

