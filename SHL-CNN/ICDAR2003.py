import sys
sys.path.append('../utils')
from get_images import get_image
import xml.etree.ElementTree as ET
import re
import numpy as np
import string
import os
from shutil import copyfile
'''
Expected structure of the IDCAR2003 directory is:
directory
|--train
|  |--char.xml
|  |--char
|  |  |--[filepaths detailed in char.xml]
|--test
|  |--char.xml
|  |--char
|  |  |--[filepaths detailed in char.xml]
'''
class ICDAR2003:
	def __init__(self,directory):
		# import training data
		self.trainfiles = [0,0]
		self.testfiles = [0,0]

		self.trainfiles[0] = self.xml_parse(directory+'/'+'train','char.xml',0)+self.xml_parse(directory+'/'+'test','char.xml',0)
		self.testfiles[0] = self.xml_parse(directory+'/'+'sample','char.xml',0)
		self.trainfiles[1] = self.xml_parse(directory+'/'+'train','char.xml',1)+self.xml_parse(directory+'/'+'test','char.xml',1)
		self.testfiles[1] = self.xml_parse(directory+'/'+'sample','char.xml',1)
		
		self.classes = (31,31)
		self.mapping = dict(zip(string.ascii_letters+string.digits,list(range(31))+list(range(31))))

	def xml_parse(self,directory,fname,mode):
		tree = ET.parse(directory+'/'+fname)
		root = tree.getroot()
		
		if mode == 0:
			regex = r'^[A-Z0-4]$'
		# elif mode == 1:
			# regex = r'^[a-z]$'
		elif mode == 1:
			regex = r'^[5-9a-z]$'

		# only want alphanumeric for this case
		imagepaths = [(directory+'/'+child.attrib['file'],child.attrib['tag'])
			for child in root if re.search(regex,child.attrib['tag'])]

		return imagepaths

	def load_data(self,dataset,size=-1):
		trainfiles = self.trainfiles[dataset][:min(size,len(self.trainfiles))]
		testfiles = self.testfiles[dataset][:min(size,len(self.testfiles))]

		train_data = np.array([get_image(file[0],(48,48)) for file in trainfiles])
		train_tags = np.array([self.mapping[file[1]] for file in trainfiles])
		test_data = np.array([get_image(file[0],(48,48)) for file in testfiles])
		test_tags = np.array([self.mapping[file[1]] for file in testfiles])

		return train_data, train_tags, test_data, test_tags

	def one_hot(self,targets,classes):
		targets = np.array([self.mapping[char] for char in targets]).reshape(-1)
		return np.eye(classes)[targets]

	def organize_data(self,newdir):
		i = 0
		for x in self.trainfiles[0]:
			if (i % 10 != 0):
				copyfile(x[0],newdir+'/1/train/'+x[1]+'/'+str(i)+'.jpg')
			else:
				copyfile(x[0],newdir+'/1/val/'+x[1]+'/'+str(i)+'.jpg')
			i+=1
		i = 0
		for x in self.testfiles[0]:
			copyfile(x[0],newdir+'/1/test/'+x[1]+'/'+str(i)+'.jpg')
			i+=1
		i = 0
		for x in self.trainfiles[1]:
			if (i % 10 != 0):
				copyfile(x[0],newdir+'/2/train/'+x[1]+'/'+str(i)+'.jpg')
			else:
				copyfile(x[0],newdir+'/2/val/'+x[1]+'/'+str(i)+'.jpg')
			i+=1
		i = 0
		for x in self.testfiles[1]:
			copyfile(x[0],newdir+'/2/test/'+x[1]+'/'+str(i)+'.jpg')
			i+=1


	
