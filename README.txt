********************************************************************************************************************************
Objective
=========
This project aims at classifying words into one of the predefined classes, given a text in which the words appear. The problem statement and data are from Stanford NLP course -cs224n. The code is my implementation of the solution to the Named Entity Recognition (NER) problem using simple RNNs and RNNs with GRU cells.


Evaluation Metric: F1-Score.
********************************************************************************************************************************

********************************************************************************************************************************
Data Overview
=============

The clases are the following:
1) Person(PER)
2) organization (ORG)
3) Location (LOC)
4) Miscellaneous (MISC)
5) Null class (O) - A class for words that can't be classified into one of the above classes.


Labeled Data looks like below:
Word    Class
=============
EU	ORG
rejects	O
German	MISC
call	O
to	O
boycott	O
British	MISC
lamb	O
.	O

Peter	PER
Blackburn	PER

BRUSSELS	LOC
1996-08-22	O

The	O
European	ORG
Commission	ORG
said	O
on	O
Thursday	O
it	O
disagreed	O
with	O
German	MISC
advice	O
to	O
consumers	O
to	O
shun	O
British	MISC
lamb	O
until	O
scientists	O
determine	O
whether	O
mad	O
cow	O
disease	O
can	O
be	O
transmitted	O
to	O
sheep	O
.	O
********************************************************************************************************************************

********************************************************************************************************************************
About Code Files
================

The code is distributed among .py files. Most of the helper classes that has code to load preprocessed data are in util.py and data_util.py files. Common model code is distributed among model.py, ner_model.py and def.py files. 

File names that begin with "q1_" implement a window-based feed-froward neural network. In this implementation, data are windowed with vectorized features and the model is suppossed to predict label for the center word in the window. File names beginning with "q2_" contain simple RNN implementation and files prefixed with "q3_" contains GRU implementation.
********************************************************************************************************************************
































