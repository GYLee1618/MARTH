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
	def __init__(self,directory,classes):
		# import training data
		self.trainfiles = self.xml_parse(directory+'/'+'train','char.xml')
		self.testfiles = self.xml_parse(directory+'/'+'test','char.xml')
		# import pdb
		# pdb.set_trace()
		self.trainfiles = self.trainfiles+self.testfiles[689:]
		self.testfiles = self.testfiles[:689]
		# pdb.set_trace()
		self.classes = 62
		self.mapping = dict(zip(string.ascii_letters+string.digits,range(62)))

	def xml_parse(self,directory,fname):
		tree = ET.parse(directory+'/'+fname)
		root = tree.getroot()
		
		# only want alphanumeric for this case
		imagepaths = [(directory+'/'+child.attrib['file'],child.attrib['tag'])
			for child in root if re.search(r'^[A-Za-z0-9]$',child.attrib['tag'])]
		return imagepaths

	def load_data(self,size=-1):
		trainfiles = self.trainfiles[:min(size,len(self.trainfiles))]
		testfiles = self.testfiles[:min(size,len(self.testfiles))]

		train_data = np.array([get_image(file[0],(48,48)) for file in trainfiles])
		train_tags = self.one_hot([file[1] for file in trainfiles],self.classes)
		test_data = np.array([get_image(file[0],(48,48)) for file in testfiles])
		test_tags = self.one_hot([file[1] for file in testfiles],self.classes)
		
		return train_data, train_tags, test_data, test_tags

	def one_hot(self,targets,classes):
		targets = np.array([self.mapping[char] for char in targets]).reshape(-1)
		return np.eye(classes)[targets]

	
