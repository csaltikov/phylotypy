# phylotypy
Naive Bayesian Classifier for 16S rRNA gene sequence data

Porting Riffomonas's CodeClub R package, phylotypr to python: https://github.com/riffomonas/phylotypr

It's been a great challenge learning how to interpret the R code into Python with minimal use of extra libraries.

It's best to clone the repository.  Run vigentte.py to see if everything works.

Training the model with the full reference database from RDP takes about 40 seconds on my MacBook Pro.

You can modify the vignette at the end to classify your own sequences. I've done this using DADA2's output files.

There's also utility.py that lets you take a fasta file of DNA seqences and process them into a dataframe for runing this classifier.

I'll make a separate vignette on how to do this and classify 16S sequence data from QIIME, DADA2, or text files.

Thanks Riffomona's for the inspiration.
