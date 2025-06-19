# ClinVar Data Analysis

## Description

This repository serves as practice for the analysis of large-scale genomic variant data from the NCBI ClinVar database in a cloud environment (JarvisLabs.ai)

The initial stage of the project focuses on establishing a robust and scalable data processing pipeline capable of handling multi-gigabyte XML datasets in a cloud computing environment. The output of this stage will be a structured, analysis-ready dataset and preliminary insights from exploratory data analysis.

## Stage 1: Goals and Objectives (Preliminary)

-   **Efficient Data Ingestion:** Develop a script to download and decompress the full ClinVar XML release directly within a cloud environment.
-   **Scalable XML Parsing:** Implement a memory-efficient streaming parser to process the ~60 GB uncompressed XML file without loading the entire dataset into memory.
-   **Data Structuring and Transformation:** Convert the relevant semi-structured XML data into a structured, columnar format (Apache Parquet) for efficient querying and analysis. Key fields to extract include:
    -   Variant details (Type, Chromosome, Position)
    -   Clinical Significance (Assertion, Review Status)
    -   Gene Information


.... For now.


## Methodology and Tech Stack

-   **Cloud Environment:** Jarvislabs.ai
-   **Data Source:** [NCBI ClinVar FTP Server](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/)
```wget -O ClinVarFullRelease_latest.xml.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/RCV_xml_old_format/ClinVarFullRelease_00-latest.xml.gz```

-   **Core Technologies:**
    -   **Language:** Python 3.11
    -   **Parsing:** `xml.etree.ElementTree.iterparse` for streaming
    -   **Data Manipulation:** `pandas`
    -   **Storage Format:** `pyarrow` for Apache Parquet
    -   **Machine Learning/Analysis:** `scikit-learn` (specifically `IncrementalPCA` for scalable dimensionality reduction)


## Decision Tree

                      +-----------------------------------------+
                      |                Root Node                |
                      |   (e.g., 10,000 variants total)         |
                      | (500 Pathogenic, 8000 Benign, 1500 VUS) |
                      +-----------------------------------------+
                                         |
                                         | Is ReviewStatus == 'reviewed by expert panel'?
                                         |
                       +-----------------+------------------+
                       | (Yes)                            | (No)
            +----------------------+           +-------------------------+
            |   Internal Node 1    |           |    Internal Node 2      |
            | (450 P, 50 B, 20 VUS)|           | (50 P, 7950 B, 1480 VUS)|
            +----------------------+           +-------------------------+
                       |                                   |
                       | Is GeneSymbol == 'BRCA1'?         | Is Chromosome == '17'?
                       |                                   |
              +--------+-------+                 +---------+----------+
              | (Yes)          | (No)            | (Yes)              | (No)
     +-----------------+  +------------+   +-------------+    +-------------------+
     |   Leaf Node A   |  | Leaf Node B|   | Leaf Node C |    |   .......         |
     |  (Pathogenic)   |  |  (Benign)  |   |    (VUS)    |    +-------------------+
     +-----------------+  +------------+   +-------------+




 **Recursive partitioning**. At each node, the algorithm considers every possible split on every feature and chooses the one that makes the resulting child nodes as **"pure"** as possible. A pure node contains data points of only a single class (e.g., 100% "Pathogenic").

"Best split" is determined by quantifying the decrease in impurity. Most common methods: Gini Impurity and Information Gain.

Let $p_k$ be proportion of data points belonging to class $k$ in a given node

#### Gini Impurity

Probability of incorrectly classifying a **randomly chosen** element in the node if it were randomly labeled according to the distribution of labels in that node.

 $G$ of a node:

$$ G = 1 - \sum_{k} (p_k)^2 $$

* If node is **perfectly pure** (all elements belong to one class, e.g., $p_1 = 1$), then $G = 1 - (1^2) = 0$
* If node has **maximum impurity** (for 2 classes, a 50/50 split where $p_1 = 0.5$ and $p_2 = 0.5$), then $G = 1 - (0.5^2 + 0.5^2) = 0.5$

The algorithm selects split that results in greatest **Gini Gain**—the largest reduction in the weighted average Gini impurity of the child nodes compared to the parent node.

#### Entropy and Information Gain

 **Entropy** from information theory. Measures level of disorder or uncertainty in a node.

The Entropy $H$ of a node:

$$ H = - \sum_{k} p_k \log_{2}(p_k) $$

* If node is **perfectly pure**, its Entropy is $H=0$
* If node has **maximum impurity** (a 50/50 split for 2 classes), its Entropy is $H=1$

**Information Gain (IG)** from a split, is the reduction in entropy from the parent to the children. The split with the **highest Information Gain** is chosen.

When splitting a set of examples $T$ on an attribute $A$:

$$ IG(T, A) = H(T) - \sum_{v \in Values(A)} \frac{|T_v|}{|T|} H(T_v) $$

Where:
* $H(T)$ is entropy of parent node
* $Values(A)$ is set of all possible values for attribute A
* $T_v$ is subset of examples for which attribute A has value $v$
* $|T_v| / |T|$ is weight of $v$-th child node
* $H(T_v)$ is entropy of $v$-th child node

### Prerequisites

-   Access to a cloud computing instance with sufficient storage (>150 GB) and RAM (>64 GB).
-   Python 3.11+ installed.
-   Required Python packages are listed in `requirements.txt`.

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/] (https://github.com/bryanlear/ClinVar.git)
    cd CardioVar-Analytics
    ```

2.  Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Execution

TBA

*(Further details on script execution will be added as they are developed.)*


