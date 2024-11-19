# IBM - MachineLearning
IBM Machine Learning Course from Coursera


## Intro
 - AI Component
 - Machine Learning: Classification, Regression, Clustering, Neural Network etc
 - Revolution in ML: Deep Learning

Data Preprocessing -> Feature Selection -> Train/Test Split -> Algorithm setup -> Model fitting -> Tuning parameters -> Prediction -> Evaluation -> Model Export

:thought_balloon: **Supervised vs Unsupervised Learning**
 - Supervised: teach the model (with data from a **labeled** dataset) then with that knowledge, it can predict unknown or future instances:
   - Classification is the process of predicting a discrete class label, or category.
   - Regression is the process of predicting a continuous value
 - Unsupervised: let the model work on its own to discover information that may not be visible (trains on the dataset, and draws conclusions on **unlabeled** data):
   - Dimension reduction / Feature Selection: reducing redundant features to make the classification easier.
   - Market basket analysis is a modeling technique based upon the theory that if you buy a certain group of items.
   - Density estimation is a very simple concept that is mostly used to explore the data to find some structure within it.
   - Clustering is used for grouping data points, or objects that are somehow similar. (for discovering structure, summarization, and anomaly detection)

## Regression Model
 - Based on the relation between dependent and independent variables
 - Simple Regression: 1 independent variable X and 1 dependent variable Y
 - Multiple Regression : multiple independent variables

Applications: Sales forecasting; Satisfaction analysis; Price estimation; Employment income; 

Regression Algorithms:
[https://www.analyticsvidhya.com/blog/2021/05/5-regression-algorithms-you-should-know-introductory-guide/]

**Simple Linear Regression**

 - residual error: the distance from the data point to the fitted regression line.
 - Minimize Mean Squared Error: The mean of all residual errors shows how poorly the line fits with the whole data set.The objective of linear regression, is to minimize this MSE equation and to minimize it, we should find the best parameters theta 0 and theta 1.

**Evaluation**
 - Training accuracy is the percentage of correct predictions that the model makes when using the test dataset.
   - a high training accuracy isn't necessarily a good thing. For instance, having a high training accuracy may result in an over-fit the data.This means that the model is overly trained to the dataset, which may capture noise and produce a non-generalized model.
 - Out of sample Accuracy: the percentage of correct predictions that the model makes on data that the model has not been trained on. 
 - K-fold cross-validation: average the accuracy of each evaluation from each fold
   - each fold is distinct, where no training data in one fold is used in another. 

 - Relative Absolute Error (RAE) is a metric expressed as a ratio normalizing the absolute error. It measures the average absolute difference between the actual and predicted values relative to the average absolute difference between the actual values and their mean.

The formula for RAE: 
**RAE = Σ|actual - predicted| / Σ|actual - mean|**

It's important to note the distinction between Relative Absolute Error (RAE) and Residual Sum of Squares (RSS):

 - Relative Absolute Error (RAE): Measures the average absolute difference between actual and predicted values relative to the average absolute difference between actual values and their mean. **(R-squared: 1-RAE** It represents how close the data values are to the fitted regression line. The higher the R-squared, the better the model fits your data. )

 - Residual Sum of Squares (RSS): Calculates the sum of the squared differences between actual and predicted values.

The formula for RSS: 
**RSS = Σ(actual - predicted)^2**


**Multiple Linear Regression**

How to estimate Theta?
 - Ordinary Least Squares (Takes long time for large dataset)
   [https://namanagr03.medium.com/deriving-ols-estimates-for-a-simple-regression-model-b3ca2b7157af]
 - Gradient Descent
   [https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21]
