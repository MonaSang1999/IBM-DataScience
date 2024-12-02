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


## Classification

- Classification is a supervised learning approach which can be thought of as a means of categorizing or classifying some unknown items into a discrete set of classes.
- Classification determines the class label for an unlabeled test case
- Classifier: binary classification (Loan Default) & multi-class classification (Which Drug for patient to use)
- Include decision trees, naive bayes, linear discriminant analysis, k-nearest neighbor, logistic regression, neural networks, and support vector machines. 

**K-Nearst Neighbors**

The K-Nearest Neighbors algorithm is a classification algorithm that takes a bunch of labeled points and uses them to learn how to label other points. This algorithm classifies cases based on their similarity to other cases. 
- This algorithm classifies cases based on their similarity to other cases.
- the distance between two cases is a measure of their dissimilarity. To calculate the similarity or conversely, the distance or dissimilarity of two data points can be done using Euclidean distance.

1. pick a value for K
2. calculate the distance of unknown case from all cases
3. search for the K-observations in the training data that are nearest to the measurements of the unknown data point.
4. predict the response of the unknown data point using the most popular response value from the K-Nearest Neighbors.

**Accuracy**
- Jaccard Index
- F1 score
- Log loss ( measures the performance of a classifier where the predicted output is a probability value between zero and one. )


**Decison Tree**

Decision trees are built by splitting the training set into distinct nodes, where one node contains all of or most of one category of the data.
- First, choose an attribute from our dataset.
 - the method uses recursive partitioning to split the training records into segments by minimizing the impurity at each step. Impurity of nodes is calculated by entropy of data in the node.
- Calculate the significance of the attribute in the splitting of the data.
- Next, split the data based on the value of the best attribute, then go to each branch and repeat it for the rest of the attributes. 


**Logistic Regression**

When to use logistic regression?
- if your data is binary: 0/1, Yes/No, True/False
- if you need probabilistic result
- when you need a linear decision boundary (we can also achieve a complex decision boundary using polynomial processing)
- if you need to understand the impact of a feature (after finding the optimum parameters, a feature X with the weight Theta one close to zero has a smaller effect on the prediction than features with large absolute values of Theta one.)

**Logistic vs. Linear Regression**

- Logistic Regression: predict the class and probability of each same belonging to the class
- Linear Regression: predict continuous values


If we use linear regression in a classification problem?
y = theta0 + theta1 X

if y < 0.5, y-hat = 0
if y >= 0.5, y-hat = 1

We can find the class, but what is the probably that the record belongs to class 0? 0.5 is a threhold which works as a step function that outputs zero or one regardless of how big or small, positive or negative the input is. As long as it's greater than 0.5, it simply equals one and vice versa, regardless of how small the value y is, the output would be zero if it is less than 0.5. 

- sigmoid function

**Minimizing the cost function of the model**
- Minimize the cost function J(Theta)
- How? Using Gradient Descent
- Gradient descent is an iterative approach to finding the minimum of a function.
- Gradient descent is a technique to use the derivative of a cost function to change the parameter values to minimize the cost or error.

Training algorithm
1. initialize the parameters randomly
2. Feed the cost function with training set, and calculate the error
3. Calculate the gradient of cost function (partial derivatives)
4. Update the weights with new values
5. go back to step 2 and feed the function again until cost is small enough
6. Predict the new customer X


**Support Vector Machine**
SVM is a supervised algorithm that classifies cased by finding a separator
1. Mapping data to a high-dimensional feature space
2. Finding a separator


## Clustering

Clustering can group data only unsupervised, based on the similarity of customers to each other. It will partition your customers into mutually exclusive groups. 

A cluster is a group of data points or objects in a dataset that are similar to other objects in the group, and dissimilar to datapoints in other clusters.

The difference between Classification and Clustering?
- Classification algorithms predict categorical classed labels. This means assigning instances to predefined classes such as defaulted or not defaulted.
- Classification is a supervised learning where each training data instance belongs to a particular class.
- Clustering is unlabeled and the process is unsupervised. we can use a clustering algorithm such as k-means to group similar customers as mentioned, and assign them to a cluster, based on whether they share similar attributes, such as; age, education, and so on.

Applications:
- Retail/Marketing:
  - Identifying buying patterns of customers
  - Recommending new books or movies to new customers
- Banking:
  - Fraud detection in credit card use
  - Identifying clusters of customers (eg.loyal)
- Insurance:
  - Fraud detection in claims analysis
  - Insurance risk of customers
- Publication:
  - Auto-categorizing news based on their content
  - Recommending similar new articles
- Medicine:
  - Characterizing patient behavior
- Biology:
  - Clustering genetic markers to identify family ties

Why Clustering:
- Exploratory data analysis
- Summary generation
- Outlier detection (Fraud detection, noise removal)
- Finding duplicates
- Pre-processing step

Different Clustering algorithms
- Partitioned-based Clustering (relatively efficient: K-Means, k-Median, Fuzzy c-Means)
- Hierarchical Clustering (Produces trees of clusters: Agglomerative, Divisive)
- Density-based Clustering (Produces arbitrary shaped clusters: DBSCAN)

**K-Means Clustering**
K-Means is a type of partitioning clustering, that is, it divides the data into K non-overlapping subsets or clusters without any cluster internal structure or labels. 


1. Initialize k (pick representative k points called centroid)
2. Distance calculation (pick different distance measures based on data)
3. Assign each point to closest centroid, creating clusters
4. Compute the new centroids for each cluster, recalculate the position of the k centroids
5. Repeat until there are no more changes

- relatively efficient on medium and large sized data sets;
- produces sphere-like clusters because the clusters are shaped around the centroids;
- its drawback is that we should pre-specify the number of clusters, and this is not an easy task. 
