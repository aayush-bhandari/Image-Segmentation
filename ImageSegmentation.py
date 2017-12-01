import cv2, numpy, sys
from sklearn.cluster import MiniBatchKMeans

filename = 'image1.jpg'
filename = sys.argv[1]
print("Input file: ",filename)
image =  cv2.imread(filename)

# calculating height, width of image
(height, width) = image.shape[:2]
print("height: ",height,"width: ",width)
#Convert to L*a*b
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# reshaping the image into a feature vector
image = image.reshape((image.shape[0] * image.shape[1], 3))

#apply k-means
k = int(sys.argv[2])
clt = MiniBatchKMeans(k)
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]

# reshape the feature vectors to images
quant = quant.reshape((height, width, 3))
image = image.reshape((height, width, 3))

# convert from L*a*b* to RGB
converted_image = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

# display the images and save on keypress
cv2.imshow("Resulted Image", numpy.hstack([converted_image]))
cv2.waitKey(0)

outfile = filename.split('.')[0]+'_result.jpg'
cv2.imwrite(outfile,converted_image)
print("Image saved as ", outfile)
