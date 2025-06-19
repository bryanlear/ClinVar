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
-   **Core Technologies:**
    -   **Language:** Python 3.11
    -   **Parsing:** `xml.etree.ElementTree.iterparse` for streaming
    -   **Data Manipulation:** `pandas`
    -   **Storage Format:** `pyarrow` for Apache Parquet
    -   **Machine Learning/Analysis:** `scikit-learn` (specifically `IncrementalPCA` for scalable dimensionality reduction)

## Getting Started

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