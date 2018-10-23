import sys
sys.path.append('../utils')
from get_images import get_image
import xml.etree.ElementTree as ET
import re
import numpy as np
import string


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
		self.trainfiles = [0,0,0]
		self.testfiles = [0,0,0]
		self.trainfiles[0] = self.xml_parse(directory+'/'+'train','char.xml',0)
		self.testfiles[0] = self.xml_parse(directory+'/'+'test','char.xml',0)
		self.trainfiles[1] = self.xml_parse(directory+'/'+'train','char.xml',1)
		self.testfiles[1] = self.xml_parse(directory+'/'+'test','char.xml',1)
		# self.trainfiles[2] = self.xml_parse(directory+'/'+'train','char.xml',2)
		# self.testfiles[2] = self.xml_parse(directory+'/'+'test','char.xml',2)
		# import pdb
		# pdb.set_trace()
		self.trainfiles[0] = self.trainfiles[0]+self.testfiles[0]
		self.testfiles[0] = None
		self.trainfiles[1] = self.trainfiles[1]+self.testfiles[1]
		self.testfiles[1] = None
		# self.trainfiles[2] = self.trainfiles[2]+self.testfiles[2]
		# self.testfiles[2] = None
		
		self.classes = (52,10)
		self.mapping = dict(zip(string.ascii_letters+string.digits,list(range(52))+list(range(10))))

	def xml_parse(self,directory,fname,mode):
		tree = ET.parse(directory+'/'+fname)
		root = tree.getroot()
		
		if mode == 0:
			regex = r'^[A-Za-z]$'
		# elif mode == 1:
			# regex = r'^[a-z]$'
		elif mode == 1:
			regex = r'^[0-9]$'

		# only want alphanumeric for this case
		imagepaths = [(directory+'/'+child.attrib['file'],child.attrib['tag'])
			for child in root if re.search(regex,child.attrib['tag'])]
		return imagepaths

	def load_data(self,dataset,size=-1):
		trainfiles = self.trainfiles[dataset][:min(size,len(self.trainfiles))]
		testfiles = None # self.testfiles[dataset][:min(size,len(self.testfiles))]

		train_data = np.array([get_image(file[0],(128,128)) for file in trainfiles])
		train_tags = self.one_hot([file[1] for file in trainfiles],self.classes[dataset])
		test_data = None #np.array([get_image(file[0],(48,48)) for file in testfiles])
		test_tags = None #self.one_hot([file[1] for file in testfiles],self.classes[dataset])

		return train_data, train_tags, test_data, test_tags

	def one_hot(self,targets,classes):
		targets = np.array([self.mapping[char] for char in targets]).reshape(-1)
		return np.eye(classes)[targets]


	
