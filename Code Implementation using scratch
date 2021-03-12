import cv2 
import os 
import glob 
import numpy as np 
import PIL
import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage.feature import local_binary_pattern,greycomatrix,greycoprops
from IPython.display import Image 
from time import time
from pylab import imshow, gray, show 
from os import path
from imutils import paths
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#Rgb values for Covid Images
img_dir = r"Enter your folder download path of covid and non covid"  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
rgb_values = [] 
for f1 in files: 
    img = cv2.imread(f1) 
    rgb_values.append(img)
    
    
print(rgb_values)

#print first image of covid
image = rgb_values[0]
#image_grey = image.convert('L')
image_array=np.array(image)
plt.imshow(image)



imagePaths = list(paths.list_images(r"C:\Users\Parth Madan\Documents\Images(CovidNon covid")) 
rawImages =[]
labels=[] 
features = []


#Extract Feature
def image_to_feature_vector(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert from BGA lo Groyscote
    return cv2.resize(image, (60, 60)).flatten()
    
    
    
#BENCHMARK Algorithm
results = []
def benchmark(clf, name): 
    print('_' * 80) 
    print("Training: ") 
    print(clf)
    t0 = time() 
    clf.fit(X_train, y_train) 
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test) 
    test_time = time() - t0
    print("test time:	%0.3fs" % test_time)

    print ( )

    score = accuracy_score(y_test, pred)
    print("Train Accuracy: %0.3f" % accuracy_score(y_train, clf.predict(X_train))) 
    print("Test Accuracy: %0.3f" % score)

    print()

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1]) 
        print("density: %f"% density(clf.coef_))

    print()
    print("classification report: ") 
    print(classification_report(y_test, pred))

    print("confusion matrix:")
    print(confusion_matrix(y_test, pred))


    print() 
    clf_descr = name
    return clf_descr, score, train_time, test_time
    
    
    
    
    start_time = time()
for (i, imagePath) in enumerate(imagePaths): 
    image = cv2.imread(imagePath)
    label=imagePath.split(os.path.sep)[-1].split(" " )[0]

    pixels = image_to_feature_vector(image)

    rawImages.append(pixels) 
    labels.append(label)
    
    print("[INFO] Processed {}/{}".format(i, len(imagePaths))) 
    print("[INFO] Processed {}/{}".format(i+1, len(imagePaths))) 
    print("--- %s seconds ---" % (time() - start_time))



imagePaths = list(paths.list_images(r"Enter image path folder for the same"))
rand = random.choices(imagePaths, k=6) 
for (i, rand) in enumerate(rand):
    img = cv2.imread(rand) 
    plt.subplot(221), imshow(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img = cv2.resize(img, (60, 60))
    label = rand.split(os.path.sep)[-1].split(". ")[0] 
    plt.subplot(222), imshow(img, 'gray') 
    plt.title(label)
    plt.show()
    
    
    
    
(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.20,random_state=109)



X = rawImages 
y=labels
X_train = trainRI 
X_test = testRI 
y_train = trainRL 
y_test = testRL



sc = StandardScaler() 
sc.fit(X_train)
X_train = sc.transform(X_train) 
X_test = sc.transform(X_test)


print("[Evaluating k-NN on raw pixel accuracy...") 
knn = KNeighborsClassifier(n_neighbors=3) 
results.append(benchmark(knn, 'k-NN on Raw Pixels'))







