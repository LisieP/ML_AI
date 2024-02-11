import numpy as np
from matplotlib import pylab as plt
import math 
#################################################################
# Import and display a signature. 
sign_full = np.loadtxt('./data/sign/sign3/sigtrain3_1.txt',comments = '%')  # Read Signature

sign = sign_full[:,:2].T

print(sign)

plt.plot(sign[0,:],sign[1,:])
plt.axis('equal')
plt.title('The original signature.')
plt.show()

################################################################
# Remove the mean
d,n = sign.shape
print(sign.shape)

mean = np.mean(sign,axis=1)[:,np.newaxis]
sign = (sign - mean)

plt.plot(sign[0,:],sign[1,:])
plt.axis('equal')
plt.title('The original signature, mean removed.(Should be centered around 0)')
plt.show()

#################################################################
# Calculate the principal directions using the SVD
u, s, vh = np.linalg.svd(sign,full_matrices=False)

# assume writing from left to right for signedness of eigenvalues
sgn = np.sign(u[0,0])
u[:,0] = sgn*u[:,0]
vh[0,:] = sgn*vh[0,:]

sgn = np.sign(u[1,1])
u[:,1] = sgn*u[:,1]
vh[1,:] = sgn*vh[1,:]

# Define Circle
ang = np.linspace(0,2*np.pi,500)
x = np.cos(ang); y = np.sin(ang)
circ = np.vstack((x,y))

# Ellipse aligned with principal directions, one standard deviation intersect.
# Note scaling of singular values so that scaled values express the standard deviations.
ell = u.dot(np.diag(s/np.sqrt(n))).dot(circ)

plt.plot(sign[0,:],sign[1,:])
plt.plot(ell[0,:],ell[1,:],'r')
plt.axis('equal')
plt.title('The signature, with ellipse.')
plt.show()

#######################################################################
# Rotating the signature
# Rotate the signature so that the principal axes coincide with the coordinate axes
new_sign = (u.T).dot(sign)
new_ell = (u.T).dot(ell)

plt.plot(new_sign[0,:],new_sign[1,:])
plt.plot(new_ell[0,:],new_ell[1,:],'r')
plt.axis('equal')
plt.title('The roated signature, with ellipse.')
plt.show()

#########################################################################
# Scale the signature
# Scale the signature, preserving the aspect ratio so that the standard deviation along the first principal axis is 1.
# Insert code to produce the image below
new_sign_scale = math.sqrt(n)/(s[0])*(new_sign)
new_ell_scale = math.sqrt(n)/(s[0])*(new_ell)

plt.plot(new_sign_scale[0,:],new_sign_scale[1,:])
plt.plot(new_ell_scale[0,:],new_ell_scale[1,:],'r')
plt.axis('equal')
plt.title('The roated and scaled signature')
plt.show()

##########################################################################
# Whiten the signature

# Now scale the signature without preserving the aspect ratio, so that the standard deviations along both principal directions equal 1.
sign_white = math.sqrt(n)*np.linalg.inv(np.diag(s)).dot(new_sign)
#sign_white = math.sqrt(n)*(new_sign)*np.linalg.inv(np.diag(s))
ell_white = math.sqrt(n)*np.linalg.inv(np.diag(s)).dot(new_ell)

plt.plot(sign_white[0,:],sign_white[1,:])
plt.plot(ell_white[0,:],ell_white[1,:],'r')
plt.axis('equal')
plt.title('The roated and whitened signature')
plt.show()
       
    