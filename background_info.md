
## Background information:

### Mathematical Procedure of Correspondence Analysis

1.  **The Contingency Table ($\mathbf{N}$)**
    The starting point is a contingency table $\mathbf{N}$ of size $I \times J$, where $I$ is the number of rows and $J$ is the number of columns. The element $n_{ij}$ represents the frequency of co-occurrence of row category $i$ and column category $j$.

2.  **The Correspondence Matrix ($\mathbf{P}$)**
    The contingency table is converted into a correspondence matrix (or probability matrix) $\mathbf{P}$ by dividing each element by the grand total, $n$.
    $n = \sum_{i=1}^{I} \sum_{j=1}^{J} n_{ij}$
    The elements $p_{ij}$ of the matrix $\mathbf{P}$ are given by:
    $$p_{ij} = \frac{n_{ij}}{n}$$

3.  **Row and Column Masses**
    The row and column marginal totals of the probability matrix $\mathbf{P}$ are calculated. These are called row masses ($\mathbf{r}$) and column masses ($\mathbf{c}$).
    * **Row masses** (a vector of size $I$):
        $$r_i = \sum_{j=1}^{J} p_{ij}$$
    * **Column masses** (a vector of size $J$):
        $$c_j = \sum_{i=1}^{I} p_{ij}$$
    Let $\mathbf{D}_r$ and $\mathbf{D}_c$ be the diagonal matrices of the row and column masses, respectively.

4.  **Expected Frequencies ($\mathbf{E}$)**
    The matrix of expected frequencies $\mathbf{E}$ is calculated under the null hypothesis of independence between the row and column variables.
    $$\mathbf{E} = \mathbf{r} \mathbf{c}^T$$
    The elements $e_{ij}$ of this matrix are $e_{ij} = r_i c_j$.

5.  **Matrix of Standardized Residuals ($\mathbf{S}$)**
    A matrix of standardized residuals $\mathbf{S}$ is computed. This matrix represents the departure of the observed data from the expected values.
    $$\mathbf{S} = \mathbf{D}_r^{-1/2} (\mathbf{P} - \mathbf{E}) \mathbf{D}_c^{-1/2}$$

6.  **Singular Value Decomposition (SVD)**
    A Singular Value Decomposition (SVD) is performed on the matrix of standardized residuals $\mathbf{S}$.
    $$\mathbf{S} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
    * $\mathbf{U}$ is the $I \times K$ matrix of left singular vectors (orthogonal).
    * $\mathbf{V}$ is the $J \times K$ matrix of right singular vectors (orthogonal), where $K = \min(I-1, J-1)$.
    * $\mathbf{\Sigma}$ is the $K \times K$ diagonal matrix of singular values ($\sigma_k$). The singular values are ordered such that $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_K > 0$.

7.  **Calculation of Principal Coordinates**
    The coordinates for the row and column categories in the new multidimensional space are calculated from the singular vectors and singular values.
    * **Standard coordinates** for the rows ($\mathbf{\Phi}$) and columns ($\mathbf{\Psi}$):
        $$\mathbf{\Phi} = \mathbf{D}_r^{-1/2} \mathbf{U}$$       $$\mathbf{\Psi} = \mathbf{D}_c^{-1/2} \mathbf{V}$$
    * **Principal coordinates** for the rows ($\mathbf{F}$) and columns ($\mathbf{G}$):
        $$\mathbf{F} = \mathbf{D}_r^{-1/2} \mathbf{U} \mathbf{\Sigma} = \mathbf{\Phi} \mathbf{\Sigma}$$       $$\mathbf{G} = \mathbf{D}_c^{-1/2} \mathbf{V} \mathbf{\Sigma} = \mathbf{\Psi} \mathbf{\Sigma}$$
    The rows of matrices $\mathbf{F}$ and $\mathbf{G}$ contain the coordinates used to plot the row and column points, respectively.

8.  **Inertia**
    The total inertia ($\mathcal{I}$) is a measure of the total variance or spread in the data. It is the sum of the squared singular values (which are the eigenvalues, $\lambda_k$, of $\mathbf{S}^T \mathbf{S}$).
    $$\mathcal{I} = \sum_{k=1}^{K} \lambda_k = \sum_{k=1}^{K} \sigma_k^2$$
    The proportion of inertia explained by each dimension $k$ is calculated as:
    $$\text{Proportion of Inertia}_k = \frac{\lambda_k}{\mathcal{I}}$$


### Combining Multiple Correspondence Analysis (MCA) with Linear Discriminant Analysis (LDA)

First, MCA is used to convert categorical variables into a set of continuous, orthogonal dimensions (principal components). Second, these continuous dimensions are used as input variables for LDA to find linear combinations that best discriminate between pre-defined groups. This approach is often called Discriminant Analysis on Principal Components.

#### Step 1: Multiple Correspondence Analysis (Recap)

The initial step is to perform a standard MCA on the set of categorical predictor variables. The key output from this stage is the matrix of principal coordinates for the individuals, often referred to as factor scores.

Let the result of the MCA be the matrix $\mathbf{F}$ of size $I \times K$, where $I$ is the number of individuals and $K$ is the number of retained dimensions (principal components) from the MCA. Each row of $\mathbf{F}$ represents an individual, and each column represents a continuous variable derived from the MCA. This matrix $\mathbf{F}$ will serve as the input for the LDA.

$$\mathbf{F} = \begin{pmatrix}
f_{11} & f_{12} & \dots & f_{1K} \\
f_{21} & f_{22} & \dots & f_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
f_{I1} & f_{I2} & \dots & f_{IK}
\end{pmatrix}$$

Additionally, we need a vector $\mathbf{y}$ of length $I$, which contains the class membership for each of the $I$ individuals. This is the dependent variable that LDA will try to predict. This variable is *not* used in the MCA step.

#### Step 2: Linear Discriminant Analysis

To find a linear combination of input variables (principal components from MCA) that maximizes the separability between the classes defined in $\mathbf{y}$.

1.  **Calculate Class-Specific and Overall Means**
    Let there be $G$ classes. Let $I_g$ be the number of individuals in class $g$.
    * The mean vector for each class $g$ in the $K$-dimensional MCA space is:
        $$\mathbf{\mu}_g = \frac{1}{I_g} \sum_{i \in \text{class } g} \mathbf{f}_i$$
        where $\mathbf{f}_i$ is the $i$-th row vector of the matrix $\mathbf{F}$.
    * The overall mean vector for the entire dataset is:
        $$\mathbf{\mu} = \frac{1}{I} \sum_{i=1}^{I} \mathbf{f}_i = \frac{1}{I} \sum_{g=1}^{G} I_g \mathbf{\mu}_g$$

2.  **Compute the Scatter Matrices**
    LDA aims to maximize the between-class scatter while minimizing the within-class scatter.

    * **Between-Class Scatter Matrix ($\mathbf{S}_B$)**: This matrix measures the variance between the different classes.
        $$\mathbf{S}_B = \sum_{g=1}^{G} I_g (\mathbf{\mu}_g - \mathbf{\mu})(\mathbf{\mu}_g - \mathbf{\mu})^T$$
        It is a $K \times K$ matrix representing the part of the total variance that is due to the differences between the group means.

    * **Within-Class Scatter Matrix ($\mathbf{S}_W$)**: This matrix measures the variance of individuals within their respective classes and sums it over all classes.
        $$\mathbf{S}_W = \sum_{g=1}^{G} \sum_{i \in \text{class } g} (\mathbf{f}_i - \mathbf{\mu}_g)(\mathbf{f}_i - \mathbf{\mu}_g)^T$$
        This is also a $K \times K$ matrix, representing the pooled covariance matrix across all groups.

3.  **Solve the Generalized Eigenvalue Problem**
   Find a transformation vector $\mathbf{w}$ that maximizes the ratio of the between-class variance to the within-class variance (Fisher's criterion):
    $$J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$
    To find the optimal $\mathbf{w}$, we solve the generalized eigenvalue problem:
    $$\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$$
    This can be rewritten as:
    $$(\mathbf{S}_W^{-1} \mathbf{S}_B) \mathbf{w} = \lambda \mathbf{w}$$

4.  **Derive the Linear Discriminants**
    Eigenvectors $\mathbf{w}_k$ of the matrix $\mathbf{S}_W^{-1} \mathbf{S}_B$. There are at most $\min(G-1, K)$ such discriminant functions.

    Eigenvector with largest eigenvalue $\lambda_k$ corresponds to the direction of maximum class separability.

5.  **Project the Data**
    Original MCA scores (matrix $\mathbf{F}$) are projected onto the new discriminant axes to obtain the discriminant scores for each individual. For a given individual with MCA scores $\mathbf{f}_i$, the score on the first linear discriminant is:
    $$z_i = \mathbf{w}_1^T \mathbf{f}_i$$

# Mathematical Description of MASCARA

MASCARA (Multivariate ASCA-Residual Analysis) is a two-stage process. The first stage involves decomposing the data according to the experimental design using ANOVA-Simultaneous Component Analysis (ASCA). The second stage involves applying Principal Component Analysis (PCA) to the residuals from the first stage.

### 1. The ASCA Model Decomposition

Let $X$ be the data matrix of size $n \times p$, where $n$ is the number of samples (observations) and $p$ is the number of measured variables.

ASCA decomposes this matrix into matrices representing the effects of the experimental factors and their interactions, plus a residual matrix. For a design with two factors, Factor A and Factor B, the model is:

$$
X = X_A + X_B + X_{AB} + E
$$

Where:
- $X_A$ is the matrix containing the variation explained by the main effect of Factor A.
- $X_B$ is the matrix containing the variation explained by the main effect of Factor B.
- $X_{AB}$ is the matrix representing the interaction effect between Factor A and Factor B.
- $E$ is the residual matrix, containing the variation not explained by the model's factors or their interactions.

The effect matrices are calculated based on the averages of the levels for each factor. For instance, the elements of $X_A$ are computed from the means of the data corresponding to each level of Factor A.

### 2. Isolation of the Residual Matrix

The residual matrix $E$ is isolated by subtracting the calculated effects from the original data matrix:

$$
E = X - (X_A + X_B + X_{AB})
$$

This matrix $E$ represents the structured variance in the data that is orthogonal to the experimental design. It serves as the input for the subsequent residual analysis.

### 3. PCA of the Residuals (The MASCARA step)

The core of MASCARA is the application of Principal Component Analysis (PCA) to the residual matrix $E$. PCA decomposes the residual matrix into scores and loadings, revealing the dominant patterns within the unexplained variation.

The PCA model for the residuals is:

$$
E = T_{res} P_{res}^T + F
$$

Where:
- $T_{res}$ is the scores matrix of size $n \times k$, where $k$ is the number of principal components. The scores represent the positions of the samples in the new principal component space. This matrix is used to find outliers, groups, or trends among the samples that are not related to the main experimental effects.
- $P_{res}$ is the loadings matrix of size $p \times k$. The loadings describe how the original variables contribute to forming the principal components. This matrix is used to understand which variables are responsible for the patterns observed in the scores.
- $F$ is the final matrix of residuals after the PCA decomposition, containing the noise not captured by the first $k$ principal components.

By analyzing the scores ($T_{res}$) and loadings ($P_{res}$), one can identify and interpret hidden phenomena, batch effects, or other systematic variations that were not part of the initial experimental design.