Introduction:
-------------
This is the dataset used in the paper (see bibtex below):

 Learning Discriminative Appearance-Based Models Using Partial Least Squares.
 W.R. Schwartz, L.S. Davis.
 XXII Brazilian Symposium on Computer Graphics and Image Processing (SIBGRAPI'2009)
 Rio de Janeiro, Brazil, October 11-14, 2009


This dataset was obtained from http://www.vision.ee.ethz.ch/~aess/iccv2007/, containing 
three video sequences used in the paper "Depth and Appearance for Mobile Scene Analysis. 
A. Ess and B. Leibe and L. Van Gool. ICCV07", so you also need to cite their paper
when using this dataset. 


Dataset:
--------
For this work we used the ground truth location of people in the video to crop each person, 
then we created a directory containing samples of each person (p0?? - p0??) for each video 
sequence. The samples in the directories have the original size, but in our experiments 
they were resized to 32x64 pixels. In our experiments, we chose one of the samples of 
each person to learn the appearance-based model and the remaining samples for 
classification (this procedure was repeated few times and the average was used). The 
results are given by the overall recognition rate.



Reference:
----------
Please reference our paper when using the data:

@inproceedings{schwartz09d,
author = {W.R. Schwartz and L.S. Davis},
booktitle = {Proceedings of the XXII Brazilian Symposium on Computer Graphics and Image Processing},
title = {Learning Discriminative Appearance-Based Models Using Partial Least Squares},
month = {Oct. 11--14, 2009},
year = {2009}
}