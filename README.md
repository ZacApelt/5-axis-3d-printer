# 5-axis-3d-printer
Development of a usable 5-axis slicer to improve the strength, model complexity, and surface finish of FDM parts

## Current progress

Basic 3-axis slicer. This generates the walls, top and bottom surface, and infill paterns. 

![alt text](image.png)

And then 5-axis slice profile generation using a wavefront propagating from the build plate. 

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)


Sliced into extrusion paths. No infill patern on this version because that's really hard. Everything is sliced with 100% infill and concentric walls. If you're doing 5-axis, you want strength, so this can kind of be justified.
Displaying every 5th layer:

![alt text](image-5.png)

Ideal orientations for each layer:

![alt text](image-6.png)