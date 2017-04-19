# -*-  encoding:utf-8  -*-
def get_test(model,ok_file,other_file,cnn,train_caffemodel):
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

	#MEAN_FILE = caffe_root+'myself/cnn_test/'+cnn+'/'+model+'/mnist_train_lmdb_mean.binaryproto'

	MEAN_FILE = '/home/jie/caffe/cnn_auto-cut_data/daxiao/5/mnist_train_lmdb_mean.binaryproto'



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

						   mean=np.load('/home/jie/caffe/cnn_auto-cut_data/daxiao/5/mnist_train_lmdb_mean.npy').mean(1).mean(1),
						   #mean = np.array([mean0,mean1,mean2]),


						   #mean = mean_tmp,


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

	test_other='/home/jie/桌面/test_data/'+other_file+'/test'
	test_ok='/home/jie/桌面/test_data/'+ok_file+'/test'
	sum_other=0
	error_other_number=0
	list_other=[]
	list_other_reliabilityp=[]
	otherp=['D']
	image_name=[]
	for root,dirs,files in os.walk(test_other):
		for file in files:
			#print file
			IMAGE_FILE = os.path.join(root,file)
			prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=False) #prediction = net.predict([caffe.io.load_image(IMAGE_FILE)],oversample=False)
			print 'image: ',file
			print 'predicted class:',prediction[0].argmax()
			sum_other=sum_other+1
			otherp.append(prediction)
			if prediction[0].argmax() == 1:

				#print ("file name is %s"%(file))

				error_other_number = error_other_number+1
				list_other.append(file)
				list_other_reliabilityp.append(prediction)
			#print("Predicted class probe argmax is #{}.".format(out['prob'].argmax()))
	#print prediction[0]
	#print 'sum_other is: ',sum_other
	#print 'error_other_number is:',error_other_number



	sum_ok=0
	error_ok_number=0
	list_ok_reliabilityp=[]
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
			if prediction[0].argmax() == 0:
				error_ok_number = error_ok_number+1
				list_ok.append(file)
				list_ok_reliabilityp.append(prediction)
			#print("Predicted class probe argmax is #{}.".format(out['prob'].argmax()))

	# print '******************************************'
	# print 'sum_other is: ',sum_other
	# print 'error_other_number is:',error_other_number
	# print 'sum_ok is: ',sum_ok
	# print 'error_ok_number is:',error_ok_number
	# print '*****************************************'
	accuracy = float((sum_ok+sum_other)-(error_ok_number+error_other_number))/float(sum_ok+sum_other)
	#print 'the accuracy is:', accuracy*100

	#FN+TP=P positive other, TN+FP=N negative ok.
	RESULT_FILE='/home/jie/桌面/test_data/result_cnn/'+cnn+'/'+model+'.txt'
	TEST_FILE='/home/jie/桌面/test_data/result_cnn/'+cnn+'/probability/'+model+'.txt'
	FN=float(error_other_number)
	FP=float(error_ok_number)
	TP=float(sum_other-FN)
	TN=float(sum_ok-FP)


	result_list=['TP:'+str(int(TP)),'FP:'+str(int(FP)),'TN:'+str(int(TN)),'FN:'+str(int(FN)),'ACC:'+str(accuracy)]

	FPR=float(FP/(FP+TN))
	result_list.append('FPR:'+str(FPR))

	FNR=float(FN/(TP+FN))
	result_list.append('FNR:'+str(FNR))

	SPE=float(TN/(TN+FP))
	result_list.append('SPE:'+str(SPE))

	SEN=float(TP/(TP+FN))
	result_list.append('SEN:'+str(SEN))


	#F1_score=2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN)))
	#result_list.append('F1_score:'+str(F1_score))
	file_object = open(RESULT_FILE, 'w')
	for i in result_list:
		file_object.writelines(i+'\n')
	file_object.writelines('ERROR_OTHER\n')
	for i in list_other:
		file_object.writelines(i+'\n')
	for i in list_other_reliabilityp:
		for j in i[0]:
			file_object.writelines(str(j)+'\t')
		file_object.writelines('\n')


	file_object.writelines('ERROR_OK\n')
	for i in list_ok:
		file_object.writelines(i+'\n')
	for i in list_ok_reliabilityp:
		for j in i[0]:
			file_object.writelines(str(j)+'\t')
		file_object.writelines('\n')

	file_object.close()

	file_object = open(TEST_FILE, 'w')
	for i in otherp:
		for j in i[0]:
			file_object.writelines(str(j)+'\t')
			# file_object.writelines(str(i[0][1])+'\n')
		file_object.writelines('\n')
	for i in okp:
		for j in i[0]:
			file_object.writelines(str(j)+'\t')
		# file_object.writelines(str(i[0][1])+'\n')
		file_object.writelines('\n')
	file_object.close()
        	#print 'prediction shape:',prediction[0].shape
		#plt.plot(prediction[0])
		#print "predicted class:%s"%(IMAGE_FILE)
		#input_image = caffe.io.load_image(IMAGE_FILE)
		#print input_image
		#prediction class


#caffe.set_mode_gpu()
#caffe.set_phase_test()

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#net.set_mean('data', mean_arr[0]) # ImageNet mean
#net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB



#caffe.set_mode_cpu()

#net = caffe.Net(MODEL_FILE,
#                PRETRAINED,
#                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1))
#net.set_mean('data',mean_arr[0])

#transformer.set_mean('data', mean_arr[0]) # mean pixel

#transformer.set_mean('data', np.load(caffe_root + 'myself/python_okornotok/okornotok_result/ilsvrc_mean.npy').mean(1).mean(1)) # mean pixel
#transformer.set_mean('data', np.load(caffe_root + 'myself/python_okornotok/okornotok_result/mean.npy').mean(1).mean(1)) # mean pixel

#mean_file = np.array([72,65,60])
#transformer.set_mean('data', mean_file)

#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB






