import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_correlated_data(n_samples=1000, rho=0.8, random_seed=42):
    """
    Generate two synthetic variables x and y with Pearson correlation approx = rho.
    Uses a bivariate normal distribution with specified covariance.
    """
    rng = np.random.default_rng(seed=random_seed)
    mean = [0, 0]
    cov_matrix = [[1.0, rho], [rho, 1.0]]
    x, y = rng.multivariate_normal(mean, cov_matrix, size=n_samples).T
    return x, y

def compute_matrices(x, y):
    """
    Compute covariance and correlation matrices of x and y.
    """
    cov_matrix = np.cov(x, y)
    corr_matrix = np.corrcoef(x, y)
    return cov_matrix, corr_matrix

def plot_scatter(x, y):
    """
    Create a scatter plot of x vs y with a regression line.
    """
    plt.figure(figsize=(6, 6))
    sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.xlabel('X variable')
    plt.ylabel('Y variable')
    plt.title('Scatter Plot with Regression Line')
    plt.grid(True)
    plt.show()

def display_matrices(cov, corr):
    """
    Display the matrices using a seaborn heatmap for better clarity.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cov, annot=True, fmt=".2f", cmap="coolwarm", ax=ax[0])
    ax[0].set_title("Covariance Matrix")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax[1])
    ax[1].set_title("Correlation Matrix")
    plt.show()

def main():
    # Step 1: Generate synthetic data
    x, y = generate_correlated_data(n_samples=1000, rho=0.8)
    
    # Step 2: Compute covariance and correlation matrices
    cov_matrix, corr_matrix = compute_matrices(x, y)
    
    print("Covariance Matrix:\n", cov_matrix)
    print("\nCorrelation Matrix:\n", corr_matrix)
    
    # Step 3: Visualize scatter and matrices
    plot_scatter(x, y)
    display_matrices(cov_matrix, corr_matrix)
    
    # Interpretation
    print("\nInterpretation:")
    print(f"- Covariance between X and Y: {cov_matrix[0,1]:.2f}")
    print("  A positive value indicates that as X increases, Y tends to increase.")
    print(f"- Pearson correlation coefficient: {corr_matrix[0,1]:.2f}")
    print("  Since it's close to 1, there's a strong positive linear relationship.")

if __name__ == "__main__":
    main()
