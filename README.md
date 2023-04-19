# riseqsar_webserver

A basic web application for serving different types of machine learning algorithms for molecular property prediction, specifically hERG and BBBP.

The web applications backend uses Docker, Flask, Gunicorn and Nginx.

Machine learning algorithms include: 
    Logistic Regression, 
    Random Forest, 
    Feedforward Neural Networks,  
    Graph Neural Networks.
    
These machine learning models are provided in separate docker containers. Deep Learning based algorithms expect GPU access with CUDA 11.3 installed.

Note that model weights are not included here neither is the training pipeline used to provide models.
