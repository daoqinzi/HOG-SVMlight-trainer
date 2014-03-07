import cv2
import numpy as np
import os
import common
import subprocess
NEG='NEG/'
POS='POS/'

def load_base(fn):
    a = np.loadtxt(fn, np.float32, delimiter=',')
    samples, responses = a[:,1:], a[:,0].astype(int)
    return samples, responses

class read_image_dir(object):
    def __init__(self,dir):
        self.listing = os.listdir(dir)
        self.n_frame = 1
        self.dir=dir

    def read(self):
        if self.n_frame < len(self.listing):
            path = self.dir + self.listing[self.n_frame]
            image = cv2.imread(path)
            image = cv2.resize(image,(40,40))
            self.n_frame = self.n_frame + 1
            print path
            print self.dir
            return image,self.dir
        else:
            return None,self.dir
class hog_features(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        # win_size= block_size= cell_size= block_stride= nbins= (40,40),(16,16),(8,8),(8,8),9,0,-1
        self.cv2hog = cv2.HOGDescriptor((40,40),(16,16),(8,8),(16,16),9,1,-1)
    def compute(self,image):
        image = cv2.resize(image,(self.x,self.y))
        image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        image_gray = cv2.equalizeHist(image_gray)
        feature_vector = self.cv2hog.compute(image_gray)#, winStride=(10,10), padding=(0,0))
        print feature_vector
        features_matrix.append(feature_vector.ravel())
        return feature_vector
jump = 0
if not(jump):
    cv2.namedWindow('image')
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                        svm_type = cv2.SVM_C_SVC,
                        C=2.67, gamma=5.383 )
    reader = read_image_dir(POS)
    cv2svm = cv2.SVM()
    features_matrix = []
    file = 'svmlight.txt'
    out_file = open(file,"w")
    hog = hog_features(50,50)
    while True:
        image,path = reader.read()
        if ((image == None) and (path == POS)) :
            reader = read_image_dir(dir=NEG)
            image,path = reader.read()
            print path
        if path == NEG and image == None:
            break
        features_vector = hog.compute(image)
        if path == POS:
            out_str = '1 '
        else:
            out_str = '-1 '
        print len(features_vector)
        for j in xrange(0,len(features_vector)):
           out_str+=(str(j+1)+':'+str(features_vector[j][0])+' ')
        out_str +='\n'
        out_file.write(out_str)
        out_str=''
    #    cv2.imshow('image',image)
        ch = cv2.waitKey(1)
        if ch == 27:
            break
    out_file.close()
    a = os.getcwd()
    a +='/./svm_learn'
    print a
    subprocess.call(['./svm_learn', "svmlight.txt", "svm_model.txt",'-C 0.01'])
    in_file = open(file,"r")
    training_data = in_file.read()
    print type(training_data)
##############
parse_string = []
weightedvector = []
file_in = 'svm_model.txt'
file_out = 'svm_weighted.txt'
out_file = open(file_out,"w")
in_file = open(file_in,"r")
for line in in_file:
  parse_string.append(line)
numfeatures = int(parse_string[7].split()[0])
print 'Number of features:',numfeatures
j = int(parse_string[9].split()[0])-1
a = float(parse_string[10].split()[0])
for n in range(0,numfeatures):
    weightedvector.append(0)
for y in range(11,len(parse_string)):
    s = parse_string[y][:-2]
    m = s.split()
    i = 1
    if (float(m[0]) > 0.01 ) or (float(m[0]) < -0.01) or i:
        print float(m[0])
        for k in range(1,numfeatures+1):
            weightedvector[k-1]+= (float(m[k].split(':')[1])*float(m[0]))
#print len(weightedvector)
#print (str(weightedvector)[1:-1])
s = parse_string[10]
m = s.split()
alpha = float(m[0])*(-1)
m = ', '+str(alpha)
s = (str(weightedvector)[1:-1])
s += m
print 'Computed Support Vector:',s
print 'Len Support Vector:',len(s.split(',')),'  Alpha:',str(alpha)
out_file.write(s)
out_file.close()
