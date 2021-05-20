# PRML_practice
Some practice for the book : Bishop - Pattern Recognition and Machine Learning
# 1.Regression
  (1) 1-dim Polynomial regression
  
      a. plot the RMS error versus M ( Amount of feature )
      
      b. plot the RMS error versus regularization coefficient(log-scale)
      
  <img src="Regression/pic/1-1.png" width="350"><img src="Regression/pic/1-2.png" width="350"><br/>
  
  (2) D-dim Polynomial regression
  
  (3) Bayesian linear regression (Gaussian basis functions)
      
      Compare the performance with different data size N
      
  <img src="Regression/pic/3a.png" width="900">
  <img src="Regression/pic/3b.png" width="900"><br/>
  
  (4) Bayesian linear regression (Sigmoid basis functions)
      
      Compare the performance with different data size N
      
<img src="Regression/pic/3c_1.png" width="900">
<img src="Regression/pic/3c_2.png" width="900"><br/>

# 2. Classification
  (1) Perceptron
      
      Implement a perceptron learning algorithm
      
  <img src="/Classification/pic/1_k.png" width="900"><br/>
  
  (2) Logistic regression
      
      Implement the Newton-Raphson algorithm
      
      a. plot the learning curve of Crossentropy error function
      
      b. plot the learning curve of Accuracy
      
  (3) Neural network
      
      Implement forward-propagation, back-propagation and stochastic gradient descent algorithm.
      
      Using Ionosphere dataset , Iris dataset and Wine dataset
      
      a. plot the learning curve of Crossentropy error function
      
      b. plot the learning curve of Accuracy
      
 <img src="/Classification/pic/3_1_iono_a.png" width="350"><img src="/Classification/pic/3_1_iono_b.png" width="350"><br/>     
 
      Activation funtion :
      
      1.NN_datasetname : hidden layer -> sigmoid
                         output layder -> softmax
                         
      2.2NN_datasetname: hidden layer -> Can choose for yourself
                         output layder -> softmax
