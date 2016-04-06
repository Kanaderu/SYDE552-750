import numpy as np

#a quick demonstration that the implemented center_surround in the feature maps can offset the constant bias

# x=np.random.normal(0,1,(1,15,15))
x=np.random.ones(0,1,(1,15,15))

# two rows = horizontal line
x[0][3][:]=10
x[0][4][:]=10

dx=np.zeros((1,15,15))
rad=3 #how large of a surround
mult=1 #how much inhibition?

for fm in np.arange(0,x.shape[0]):
    for i in np.arange(rad,x.shape[1]-rad):
        for j in np.arange(rad,x.shape[2]-rad):
            dx[fm][i][j]=int(-mult*rad*np.average(x[fm][i-rad:i+rad+1][:,j-rad:j+rad+1])+x[fm][i][j])
            
print x.astype(int)
print (x+dx).astype(int)