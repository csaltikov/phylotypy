# phylotypy
Naive Bayesian Classifier for 16S rRNA gene sequence data

Porting Riffomonas's CodeClub R package, phylotypr to python: https://github.com/riffomonas/phylotypr

It's been a great challenge learning how to interpret the R code into Python with minimal use of extra libraries.

It's best to clone the repository.  Run vigentte.py to see if everything works.

If it does then you can modify the vignette to classify your own sequences.
I've done this using DAD2's output files.  I made a utility that lets you process the DADA2 file and run this classifier
I'll make a separate tutorial on how to classify 16S sequence data from QIIME, DAD2, and other tools. 
