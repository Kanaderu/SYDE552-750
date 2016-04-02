# Peter Duggins
# SYDE 552/750
# Final Project
# Script to generate, label, and manipulate simple lines, for input to nengo/keras attention CNN

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

pixels_x=64
pixels_y=64
N=50000
images=[]
labels=[]

width_mean=2
width_sigma=2
intensity_mean=0.7
intensity_sigma=0.2
angle_sigma=5
filter_sigma=2
filter_unif=5
noise_sigma=0.05


for n in range(N):
	if np.mod(n,100)==0:
		print '%s/%s images generated' %(n,N)
	img=np.zeros((pixels_x, pixels_y))
	label=np.random.random_integers(low=0,high=2) #0=horizontal, 1=vertical, 2=diagonal
	#make the spanning vertical or horizontal lines
	center=np.random.uniform(low=0,high=pixels_x)
	width=np.random.normal(loc=width_mean,scale=width_sigma)
	loc_min=np.floor(max(center-width/2,0))
	loc_max=np.ceil(min(center+width/2,pixels_x))
	line_r=[loc_min,loc_max] #row
	line_c=[loc_min,loc_max] #column
	line_i=np.random.normal(loc=intensity_mean,scale=intensity_sigma) #pixel intensity
	for i in range(pixels_x):
		for j in range(pixels_y):
			if label == 0 and j>=np.min(line_r) and j <= np.max(line_r):
				img[j][i]=line_i
			if label == 1 and i>=np.min(line_c) and i <= np.max(line_c):
				img[j][i]=line_i
			if (label == 2 and i==j): # or (label == -1 and i==pixels_x-j) #gives X shape
				img[j][i]=line_i

	#apply cropping (changes imagesize = bad)
	# img=img[pixels_x/4:-pixels_x/4, pixels_y/4:-pixels_y/4]

	#apply rotation
	angle=np.random.normal(loc=0,scale=angle_sigma)
	img = ndimage.rotate(img,angle,reshape=False)

	#apply filter
	img=ndimage.gaussian_filter(img,sigma=filter_sigma) #gaussian
	# img=ndimage.uniform_filter(img, size=filter_unif) #uniform

	#apply white noise
	img=img+np.random.normal(loc=0, scale=noise_sigma, size=(pixels_x,pixels_y))

	images.append(img)
	labels.append(label)
	# plt.imshow(img, cmap='gray',vmin=0, vmax=1)
	# plt.show()

np.save('lines_data',np.array(images))
np.save('lines_labels',np.array(labels))