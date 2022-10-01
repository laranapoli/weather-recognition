# Weather Recognition Neural Network
A data science project to automate climate recognition from images.
![Weather Recognition](https://miro.medium.com/max/900/1*QBJ6XiG4NT2NASMu8XvSIQ.png)

## Table of contents
* [General Info](#general-info)
* [Technologies](#technologies)
* [Status](#status)
* [Sources](#sources)

## General Info
Automated weather recognition has important application value in traffic safety, automobile auxiliary driving and meteorology.

This project aims to implement a neural network from scratch using numpy. Python modules were developed in order to choose an architecture and train a neural network to identify weather within four classes: cloudy, rain, shine and sunrise. 

My purpose was to learn the fundamentals and acquire practical knowledge. By creating from scratch this neural network, I had an opportunity to use softmax function (generally used in output layer for multiclass problems). Outputs are probabilities of every possible class, wich sums up to 1. The predicted label is the one with the highest probability.
![Neural Network Architecture with Softmax](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/04/Screenshot-from-2021-04-01-17-25-02.png)

## Technologies
- Python 3.7.6
- numpy 1.21.2
- matplotlib 3.2.1
- seaborn 0.11.2
- pickle 4.0
- pandas 1.3.2
- opencv-python 4.6.0.66
- Keras-Preprocessing 1.1.2

## Sources
This project was inspired by Andrew Ng's course ["Neural Networks and Deep Learning"](https://www.coursera.org/learn/neural-networks-deep-learning).

Shout out to [Harrison Kinsley](https://github.com/Sentdex) for providing video content of his book ["Neural Networks from Scratch in Python"](https://nnfs.io/).