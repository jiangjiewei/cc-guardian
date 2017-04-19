# -*-  encoding:utf-8  -*-
def get_result_feature(model,ok_file,other_file,cnn,train_caffemodel):
	import numpy as np
	import matplotlib.pyplot as plt
	import os,types

	caffe_root = '/home/jie/caffe/caffe-master/'
	import sys
	sys.path.insert(0,caffe_root+'python')

	import caffe


	###/home/jie/桌面/cnn_test/cnn/ok-8/caffe_alexnet_train_iter_2000.caffemodel
	###/home/jie/桌面/cnn_test/cnn/ok-8/deploy.prototxt


	#MODEL_FILE = '/home/jie/桌面/cnn_test/cnn/ok-8/caffe_alexnet_train_iter_2000.caffemodel'
	#PRETRAINED = '/home/jie/桌面/cnn_test/cnn/ok-8/deploy.prototxt'



	MODEL_FILE = caffe_root+'myself/cnn_test/'+cnn+'/deploy.prototxt'
	PRETRAINED = caffe_root+'myself/cnn_test/'+cnn+'/'+model+'/'+train_caffemodel+''

	#TEST_FILE = '../data/liris-accede/test_arousal.txt'

	MEAN_FILE = caffe_root+'myself/cnn_test/'+cnn+'/'+model+'/mnist_train_lmdb_mean.binaryproto'



	# Open mean.binaryproto file

	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(MEAN_FILE , 'rb').read()
	blob.ParseFromString(data)
	mean_arr = caffe.io.blobproto_to_array(blob)
	#mean1 = mean_arr[0]
	#mean_test=mean_arr[0].mean(1)
	mean_test1 = mean_arr[0].mean(1).mean(1)
	mean_zero=[0,0,0]
	mean_tmp=np.array(mean_zero)
	mean_tmp[0]=int(round(mean_test1[0]))
	mean_tmp[1]=int(round(mean_test1[1]))
	mean_tmp[2]=int(round(mean_test1[2]))


	# Initialize NN
	# Initialize NN
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,


						   image_dims=(256,256),

						   #mean=np.load(caffe_root + 'myself/python_okornotok/okornotok_result/mnist_train_lmdb_mean.npy').mean(1).mean(1),
						   #mean = np.array([mean0,mean1,mean2]),
						   mean = mean_tmp,
						   #mean= mean_arr[0].mean(1).mean(1),
						  # mean=np.load(caffe_root + 'myself/python_okornotok/okornotok_result/mnist_train_lmdb_mean.npy').mean(1).mean(1),

						   #input_scale=255,



						   raw_scale=255,
						   channel_swap=(2,1,0)
							)

	net.blobs['data'].reshape(1,3,224,224)
	#test_file1 = "/home/jie/caffe/caffe-master/myself/python_okornotok/okornotok5-5/other-5/test"
	#test_file2 = "/home/jie/caffe/caffe-master/myself/python_okornotok/okornotok5-5/ok-5/test"
	#test_file3 = "/home/jie/caffe/caffe-master/myself/python_okornotok/okornotok/train/ok"

	test_other='/home/jie/桌面/test_data/'+other_file+'/train'
	test_ok='/home/jie/桌面/test_data/'+ok_file+'/train'
	feature_other='/home/jie/桌面/test_data/result_cnn/'+cnn+'/'+model+'_otherfeature.txt'
	feature_ok='/home/jie/桌面/test_data/result_cnn/'+cnn+'/'+model+'_okfeature.txt'

	sum_other=0
	error_other_number=0
	list_other=[]
	otherp=['D']
	for root,dirs,files in os.walk(test_other):
		for file in files:
			#print file
			IMAGE_FILE = os.path.join(root,file)
			prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=False) #prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=False)
			#print 'image: ',file
			print 'predicted class:',prediction[0].argmax()
			#features = [features]
			#out1 = net.blobs['prob'].data[0]
			#out2 = net.blobs['fc8'].data[0]
			#print("Predicted class is #{}.".format(out['prob'][0].argmax()))
			sum_other=sum_other+1
			otherp.append(prediction)
			features = net.blobs['fc7'].data[0]
			#featurep.append(features)

			file_object = open(feature_other, 'a')
			for i in features:
				#for j in i:
					file_object.writelines(str(i)+' ')
					# file_object.writelines(str(i[0][1])+'\n')
			file_object.writelines('\n')
			file_object.close()

			if prediction[0].argmax() == 1:
				error_other_number = error_other_number+1
				list_other.append(file)
			#print("Predicted class probe argmax is #{}.".format(out['prob'].argmax()))
	#print prediction[0]
	#print 'sum_other is: ',sum_other
	#print 'error_other_number is:',error_other_number



	sum_ok=0
	error_ok_number=0
	list_ok=[]
	okp=['N']
	for root,dirs,files in os.walk(test_ok):
		for file in files:
			#print file
			IMAGE_FILE = os.path.join(root,file)
			prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=False) #prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=False)
			print 'predicted class:',prediction[0].argmax()
			sum_ok=sum_ok+1
			okp.append(prediction)
			features = net.blobs['fc7'].data[0]
			#featurep.append(features)
			file_object = open(feature_ok, 'a')
			for i in features:
				file_object.writelines(str(i)+' ')
				# file_object.writelines(str(i[0][1])+'\n')
			file_object.writelines('\n')
			file_object.close()

			if prediction[0].argmax() == 0:
				error_ok_number = error_ok_number+1
				list_ok.append(file)
			#print("Predicted class probe argmax is #{}.".format(out['prob'].argmax()))

	print '******************************************'
	print 'sum_other is: ',sum_other
	print 'error_other_number is:',error_other_number
	print 'sum_ok is: ',sum_ok
	print 'error_ok_number is:',error_ok_number
	print '*****************************************'
