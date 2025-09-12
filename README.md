Exercise: Create Vectors and Matrices and Perform Basic Arithmetic
Task Ideas:
Create two vectors of size 4 with any numbers; add and subtract them.

Multiply a vector by a scalar.

Create two 3×3 matrices and add them.

Multiply a matrix by a scalar.

Use indexing to print:

The second element of a vector

The element in the first row, third column of a matrix

Slice the first two elements of a vector and the first two rows of a matrix.

# Tuesday 29th July Exercise
Multiply two 2×2 matrices and check your answer by hand and with NumPy.
Create an identity and zero matrix in NumPy; demonstrate that A x I = A and A + Z = A
Apply a scaling and rotation transformation to a set of 2D points; visualize with Matplotlib.

Experiment with changing the transformation matrix to see effects (try reflection, shearing, etc.).


# Wednesday 30th July Exercise
Create several matrices—check and interpret their rank.

Calculate and interpret eigenvalues/eigenvectors for simple 2x2 matrices.
Visualize how a matrix transformation stretches along eigenvector directions.

# Monday 04th August Exercise
1. Write a Python function to compute the numerical derivative of any user-supplied function at any point
2. Use your function to compute the derivative of f(x) = 3x^3 + 2x at several points.
3. Show how small vs large dx affects the accuracy of your numerical derivative
-Compute and print the derivative at x = 1 for different values of dx
-Observe how the numerical approximation improves or becomes unstable.
4. Create a plot showing both a function and its numerical derivative on the same graph

# Tuesday 5th August Exercise 

Compute Gradients for Multivariable Functions — Given function: f(x, y) = 3x^2y + 2*y^3.
(a) Compute analytically the partial derivatives ∂f/∂x and ∂f/∂y. 
(b) Write a Python function for f(x, y).
(c) Implement a function to compute the partial derivatives numerically at (x=1, y=2).

# Wednesday 6th August 2025 Exercise
Implement gradient descent to minimize the function:
f(x) = x^2 + 2*x + 1
(the minimum is at x = -1)

# Tuesday 12th August 2025 Exercise
Simulate rolling two dice 10,000 times.

Find the probability distribution for the sum of the two dice.

Plot the probability distribution and identify the most likely sums.

# Wednesday 13th August 2025 Exercise
Generate two synthetic variables with different levels of correlation.

Compute and interpret their covariance and correlation matrices.

Create scatter plots for visual inspection.

# Monday 18th August 2025 Exercise
Simulate your own population data (change mean, std).

Draw samples of different sizes (10, 100, 1000).

Calculate the sample mean and 95% confidence interval for each sample.

Observe how the confidence interval changes with sample size.

# Tuesday 19th August 2025 Exercise
Simulate your own sample data with a different mean.

Test if the sample mean differs significantly from a chosen population mean.

Try varying sample size and observe effect on significance.

# Wednesday 20th August 2025 Exercise
Extend this code (linear_regression.py) to:

Calculate residuals (difference between actual and predicted values).

Compute R-squared (goodness of fit).

Try fitting using multiple features (x with two or more columns).

Interpret coefficients in context of real-world data.

# Machine Learning Test Project
Project Tasks
1. Data Loading and Exploration
Load the Iris dataset from scikit-learn.

Display basic information: shape, feature names, and first 5 rows of data.

2. K-Means Clustering
Perform K-Means clustering with 3 clusters (n_clusters=3).

Print cluster centers and inertia.

Calculate and print the silhouette score.

Visualize the clusters and centroids using the first two features in a scatter plot.

3. Principal Component Analysis (PCA)
Apply PCA to reduce the Iris dataset dimensions from 4 to 2.

Print the explained variance ratio of the two components.

Visualize the data on the first two principal components scatter plot, coloring points by their true species labels.

4. Hierarchical Clustering
Perform hierarchical agglomerative clustering on the Iris data using Ward linkage.

Plot the dendrogram to show cluster merges.

Cut the dendrogram at an appropriate distance to form clusters, assign cluster labels.

Print these cluster assignments.

5. Comparative Visualization
Create a 2x2 subplot figure with:

K-Means clusters (scatter plot with centroids).

PCA projection colored by true labels.

Dendrogram from hierarchical clustering.

Hierarchical clustering flat cluster assignments scatter plot.


Bonus Challenge
Experiment with different values of K and linkage methods (single, complete, average) for hierarchical clustering.

Generate elbow plots and silhouette scores for K-Means with varying K.

Report which configurations produce the best clusters and why.
