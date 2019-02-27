# python3 install -m pip install Pillow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt 
import os

def run_train():
	print("[INFO] Started")

	#print(K.image_data_format())

	# path to the model weights files
	weights_path = './weights/example1_cnn.h5'

	# path to datasets
	train_data_dir = 'dataset/train'
	validation_data_dir = 'dataset/validation'

	#dimension of images
	img_width, img_lenght = 15, 15

	# number of samples
	#nb_train_samples = 5100
	nb_train_samples = 2700
	#nb_validation_samples = 1520
	nb_validation_samples = 840

	# settings of training step
	epochs = 100
	batch_size = 20

	# preprocessing data
	#  augmentation configutarion for training
	train_datagen = ImageDataGenerator( rescale= 1./255)#,
			#rescale = 1./255,
			#shear_range = 0.2,
			#zoom_range = 0.2)
			#horizontal_flip = True)
	# augmentation configuration for validation
	validation_datagen = ImageDataGenerator( rescale = 1./255)

	print("[INFO] Loading data")

	# loading data
	train_data = train_datagen.flow_from_directory(train_data_dir, color_mode='grayscale',target_size=(img_width, img_lenght), batch_size= batch_size, classes=['buy','hold','sell'], class_mode='categorical', shuffle=False)
	validation_data = validation_datagen.flow_from_directory(validation_data_dir,color_mode='grayscale', target_size=(img_width, img_lenght), batch_size=batch_size, classes=['buy','hold','sell'], class_mode='categorical', shuffle=False)


	# building CNN

	# TensorFlow is channels_last
	input_shape = (img_width, img_lenght, 1)

	cnn = Sequential()
	cnn.add( Conv2D(32, (3,3), padding='same', input_shape= input_shape) ) # 16 filters, 3x3
	#cnn.add(Activation('relu'))
	#cnn.add(MaxPooling2D(pool_size=(2,2)))

	cnn.add(Conv2D(64, (3,3), padding='same')) # 32 filters, 5x5
	cnn.add(Activation('relu'))

	cnn.add(MaxPooling2D(pool_size=(2,2)))

	cnn.add(Dropout(0.25))

	cnn.add(Flatten())
	cnn.add(Dense(128))
	cnn.add(Activation('relu'))

	#cnn.add(Dense(64))
	#cnn.add(Activation('relu'))

	cnn.add(Dropout(0.5))

	cnn.add(Dense(3)) # 3 classes
	cnn.add(Activation('softmax'))

	cnn.compile(loss='categorical_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
	print(cnn.summary())

	tensorboard = TensorBoard(batch_size=batch_size)
	print("[INFO] Training...")
	H = cnn.fit_generator(train_data, steps_per_epoch= nb_train_samples //
						batch_size, epochs=epochs, validation_data=validation_data, validation_steps= nb_validation_samples //
						batch_size, verbose=0, callbacks=[tensorboard])

	# saving wieghts
	os.makedirs("weights",exist_ok=True)
	cnn.save_weights(weights_path)
	# saving model
	cnn.save("cnn_model")

	#plotting convergence curve
	plt.figure(figsize=[8,6])
	plt.plot(H.history['loss'], 'r', linewidth=3.0)
	plt.plot(H.history['val_loss'], 'b', linewidth=3.0)
	plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
	plt.xlabel('Epochs ', fontsize=16)
	plt.ylabel('Loss ',  fontsize=16)
	plt.title('Loss Curve', fontsize=22)

	plt.savefig('loss.jpg')

	# Accuracy curves
	plt.figure(figsize=[8,6])
	plt.plot(H.history['acc'], 'r', linewidth=3.0)
	plt.plot(H.history['val_acc'], 'b', linewidth=3.0)
	plt.legend(['Traning Accuracy', 'Validation Accuracy'], fontsize=18)
	plt.xlabel('Epochs', fontsize=16)
	plt.ylabel('Accuracy', fontsize=16)
	plt.title('Accuracy Curves', fontsize=22)

	plt.savefig('accuracy.jpg')

if __name__ == "__main__":
	folds = ["fold_1", "fold_2","fold_3","fold_4","fold_5", "fold_6",\
	         "fold_7", "fold_8","fold_9","fold_10"]
	root_dir = os.getcwd()
	for fold in folds:
		#os.chdir(root_dir+"/experimento/"+fold)
		os.chdir(root_dir+"/experiment3/"+fold)
		print(os.getcwd())
		run_train()
	print("**** Finish ****")

