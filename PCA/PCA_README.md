# PCA using first Principles
Essentially, PCA finds lower-dimensional subspaces that describe the essential properties of our data. This allows one to project the data onto these lower-dimensional spaces.

One interesting task is building a signature verification system. One approach to do that is constructing a probabilistic model from a number of training signatures of each signatory (each individual in a database). This probabilistic model should capture and describe the natural variations between different signatures for each signatory. This of course has to be learned from the training signatures.

But signatures also vary as to the origin of whatever coordinate system we use, the size and (in-plane) rotation. It might be possible to also ask the system to recognize signatures at different positions, sizes and rotations, as the same signature. This however, vastly complicates the system and will demand a very large training set representing all these different situations. Instead, it is much easier to do some pre-processing normalizing the signatures.

Although it will fail on some signatures, PCA is a valuable, simple tool for doing just this. At the same time it provides a great illustration of the underlying mechanism of the singular value decomposition.

These signatures were captured on a digitising tablet, capturing the  𝑥 and  𝑦 coordinates, the pen pressure, the pen angle(tilt) and the pen direction. For now, we will only use the  𝑥 and  𝑦 coordinates, which are the first two features in the provided data.

# Logic flow
The math is not explained in depth heare but easy to follow in the code
## 1. Removing the mean
The first step is to remove the mean of the signature. This centers the signature, i.e. normalizes it with respect to position.

## 2. Calculating the principal components
We calculate the principal components using the SVD. We illustrate the principal directions as well as the one standard deviation by calculating the one standard deviation ellipse, aligned along the principal directions.

## 3. Rotating the signature
Rotate the signature so that the principal axes coincide with the coordinate axes by using components of SVD

## 4. Scaling of signature
Scale the signature, preserving the aspect ratio so that the standard deviation along the first principal axis is 1.

## 5. Whitening of signature
Scale the signature without preserving the aspect ratio, so that the standard deviations along both principal directions equal 1.