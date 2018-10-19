import xml.etree.ElementTree as ET

path = "data/ICDAR2003_RRTL_sample/"


class taggedrect:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.offset = 0

class Image:
    def __init__(self):
        self.name = ""
        self.x_res = 0
        self.y_res = 0
        self.tagged = []
img_dict = {}

def parseXML(xml_file):
    """
    Parse XML with ElementTree, feed into dictionary
    """
    tree = ET.ElementTree(file=xml_file)
    root = tree.getroot()

    for tag in root:
        info = list(tag)
        if tag.tag == 'image':
            node = Image()
            for child in info:
                if child.tag == 'imageName':
                    node.name = child.text
                if child.tag == 'resolution':
                    node.x_res = child.attrib['x']
                    node.y_res = child.attrib['y']
                if child.tag == 'taggedRectangles':
                    for rect in child:
                        rectangles = taggedrect()
                        rectangles.x = rect.attrib['x']
                        rectangles.y = rect.attrib['y']
                        rectangles.width = rect.attrib['width']
                        rectangles.height = rect.attrib['height']
                        rectangles.offset = rect.attrib['offset']
                        node.tagged.append(rectangles)
            img_dict[node.name] = node     
    ##work in progress, loaded nodes into dictionary
    for keys in img_dict:
        print (keys)
if __name__ == "__main__":
    parseXML(path+'locations.xml')

