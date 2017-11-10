import cv2
import numpy as np 
from glob import glob 
import argparse
from helpers import *
from matplotlib import pyplot as plt 
import os
import warnings
warnings.filterwarnings("ignore")

class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.test_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def trainModel(self):
        """
        This method contains the entire module 
        required for training the bag of visual words model

        Use of helper functions will be extensive.

        """



        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
        # extract SIFT Features from each image
        label_count = 0 
        noneCount = 0
        for word, imlist in self.images.iteritems():
            self.name_dict[str(label_count)] = word
            print "Computing Features for ", word
            for im in imlist:
                #cv2.imshow("im", im)
                #cv2.waitKey()
                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.im_helper.features(im)
                if des==None:
                    noneCount+=1
                else:
                    self.descriptor_list.append(des)
               

            label_count += 1
        #print len(self.descriptor_list[1])
        
        self.trainImageCount = self.trainImageCount - noneCount
        # perform clustering
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        # self.bov_helper.plotHist()
 

        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)


    def recognize(self,test_img,    test_image_path=None):

        """ 
        This method recognizes a single image 
        It can be utilized individually as well.


        """

        kp, des = self.im_helper.features(test_img)

        # generate vocab for test image
        vocab = np.array( [ 0 for i in range(self.no_clusters)])
        # locate nearest clusters for each of 
        # the visual word (feature) present in the image
        
        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)

        for each in test_ret:
            vocab[each] += 1

        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)

        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb



    def testModel(self):
        """ 
        This method is to test the trained classifier

        read all images from testing path 
        use BOVHelpers.predict() function to obtain classes of each image

        """

        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)
        pred = np.array([])
        predictions = []
        label_count = 0
        for word, imlist in self.testImages.iteritems():
            print "processing " ,word
            for im in imlist:
                cl = self.recognize(im)
                pred = np.append(pred,cl)
                self.test_labels = np.append(self.test_labels, label_count)
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))]
                    })
            label_count += 1

        #print predictions
        #for each in predictions:
        #    cv2.imshow(each['object_name'], each['image'])
        #    cv2.waitKey()
        #    cv2.destroyWindow(each['object_name'])
           
        self.bov_helper.evaluate(self.test_labels,pred)
		
    def print_vars(self):
        pass
    

if __name__ == '__main__':

    # parse cmd args
    parser = argparse.ArgumentParser(
            description=" Bag of visual words example"
        )
    parser.add_argument('--train_path', action="store", dest="train_path", required=False)
    parser.add_argument('--test_path', action="store", dest="test_path", required=False)

    args =  vars(parser.parse_args())
   

    
    bov = BOV(no_clusters=100)

    # set training paths
    bov.train_path = 'images/train/'     #args['train_path'] 
    # set testing paths
    bov.test_path = 'images/test/'      #args['test_path'] 
    # train the model
    bov.trainModel()
    # test model
    bov.testModel()
	