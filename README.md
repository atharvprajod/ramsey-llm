# Ramsey Theory for Concept Subspaces in Language Model Embeddings

This repository contains a validation script that implements a theoretical framework for understanding the emergence of linear structures in language model embeddings, leveraging Ramsey theory. The script is designed to validate the Geometric Regularity Hypothesis (GRH) and Ramsey bounds as discussed in the accompanying paper.

## Overview

The main components of the script include:

- **ValidationConfig**: A configuration class to set parameters for the validation process, including the embedding dimension, subspace dimension, approximation tolerance, number of samples, and number of trials.
  
- **RamseyValidator**: The main class responsible for validating the GRH predictions and Ramsey bounds. It includes methods for:
  - Calculating the theoretical Ramsey number \( R(k, \epsilon) \).
  - Finding a monochromatic clique in the embeddings.
  - Performing synthetic verification to compare observed Ramsey thresholds against theoretical predictions.

- **ModelValidator**: An extended validator for testing real language models. It includes methods for:
  - Extracting embeddings from a given model.
  - Computing subspace coherence to compare GRH and LRH predictions.
  - Conducting interpretability analysis to evaluate the practical utility of Ramsey subspaces.

## Requirements

To run the script, you will need the following Python packages:

- numpy
- torch
- scipy
- scikit-learn
- tqdm

You can install the required packages using pip:

```bash
pip install numpy torch scipy scikit-learn tqdm
```

## Usage

1. **Synthetic Verification**: The script can be run to perform synthetic verification of the Ramsey bounds. The `main` function demonstrates how to set up the validation configuration and execute the synthetic verification.

2. **Model Validation**: For testing with real language models, you can uncomment the relevant sections in the `main` function and provide your model and tokens.

### Example

```python
def main():
    # Example usage
    config = ValidationConfig(
        dimension=768,  # typical embedding dimension
        k=3,            # looking for 3D concept subspaces
        epsilon=0.1,    # 10% approximation tolerance
        num_samples=10000
    )
    
    # Phase 1: Synthetic verification
    validator = RamseyValidator(config)
    synthetic_results = validator.synthetic_verification()
    print("Synthetic Results:", synthetic_results)
    
    # For testing with real models:
    """
    model_validator = ModelValidator(config)
    
    # Phase 2: Model comparison
    model = load_your_model()  # e.g., GPT-2, BERT, etc.
    tokens = get_test_tokens()
    comparison_results = model_validator.compare_hypotheses(model, tokens)
    print("Comparison Results:", comparison_results)
    
    # Phase 3: Interpretability
    embeddings = model_validator.extract_embeddings(model, tokens)
    interp_results = model_validator.interpretability_analysis(embeddings)
    print("Interpretability Results:", interp_results)
    """
```

## Conclusion

This script provides a framework for validating the theoretical predictions regarding the emergence of linear structures in high-dimensional embedding spaces. By leveraging Ramsey theory, it aims to enhance our understanding of the geometric properties of language model embeddings.