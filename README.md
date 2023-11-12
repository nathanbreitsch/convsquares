# convsquares
A pytorch / python implementation of marching squares using conv2d as the workhorse

## Why?
This algorithm is useful for finding the points around the permimeter of an instance segmentation mask.  It is available in scikit-image as `find_contours` but requires converting to a numpy array.  This can be annoying if 
you want to keep the data on its device.

## How?
A big part of marching squares involves iterating over pixels to find and classify boundary points.  This can be achieved by convolving the following 2x2 kernel
```
[[1, 2],
 [4, 8]]
```
and using the resulting values to look up relative coordinate edges.

The process of joining these edges into contours is still just a for loop, but usually there are few enough edges in the contour that this step is negligible.

## Current Limitations
- Does not apply interpolation
- Is not differentiable (could it be?)
- Cannot be compiled using dynamo or torchscript (could it be?)
