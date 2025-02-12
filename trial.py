import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict, Optional, Union
from transformers import (
    GPTNeoXModel, 
    GPTNeoXTokenizerFast,  # Use the Fast tokenizer
    LlamaModel, 
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)

from dataclasses import dataclass
import logging
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict, Optional
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from tqdm import tqdm
import logging
from dataclasses import dataclass

@dataclass
class ValidationConfig:
    dimension: int
    k: int  # subspace dimension
    epsilon: float  # approximation tolerance
    num_samples: int  # number of embeddings to sample
    num_trials: int = 10
    confidence_level: float = 0.95
    
class RamseyValidator:
    """Main class for validating GRH predictions and Ramsey bounds."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("RamseyValidator")
        logger.setLevel(logging.INFO)
        return logger
        
    def theoretical_ramsey_bound(self) -> float:
        """Calculate theoretical Ramsey number R(k,ε) from the paper."""
        c = 1.0  # universal constant from the paper
        d = self.config.dimension
        k = self.config.k
        eps = self.config.epsilon
        
        # Calculate in log space to avoid overflow
        # log(R(k,ε)) = log(k+1) + (k^2(d-k)) * log(c/ε)
        log_result = np.log(k + 1) + (k * k * (d - k)) * np.log(c / eps)
        
        # Return the log of the bound to avoid overflow
        return log_result
    
    def find_monochromatic_clique(self, embeddings: np.ndarray) -> Tuple[np.ndarray, float]:
        """Find a (k+1)-clique with differences well-approximated by a k-dim subspace."""
        n_samples = len(embeddings)
        best_clique = None
        best_error = float('inf')
        
        # Sample candidate cliques
        for _ in range(self.config.num_trials):
            # Randomly sample k+1 points
            indices = np.random.choice(n_samples, self.config.k + 1, replace=False)
            clique = embeddings[indices]
            
            # Compute pairwise differences
            diffs = []
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    diffs.append(clique[i] - clique[j])
            diffs = np.stack(diffs)
            
            # Find best k-dimensional subspace approximating differences
            pca = PCA(n_components=self.config.k)
            pca.fit(diffs)
            
            # Compute approximation error
            projected = pca.inverse_transform(pca.transform(diffs))
            error = np.max(np.linalg.norm(diffs - projected, axis=1))
            
            if error < best_error:
                best_error = error
                best_clique = clique
                
        return best_clique, best_error
    
    def synthetic_verification(self) -> Dict[str, float]:
        """Phase 1: Verify Ramsey bounds with synthetic data."""
        self.logger.info("Running synthetic verification...")
        
        log_theoretical_R = self.theoretical_ramsey_bound()
        observed_Rs = []
        
        for _ in range(self.config.num_trials):
            # Generate random embeddings
            embeddings = np.random.normal(0, 1, 
                                        (self.config.num_samples, self.config.dimension))
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Find monochromatic clique
            _, error = self.find_monochromatic_clique(embeddings)
            
            # Record minimum N needed for ε-approximation
            if error <= self.config.epsilon:
                observed_Rs.append(self.config.num_samples)
                
        # Statistical validation
        if len(observed_Rs) > 1:  # Need at least 2 samples for std calculation
            observed_R = np.mean(observed_Rs)
            log_observed_R = np.log(observed_R)
            log_observed_Rs = [np.log(r) for r in observed_Rs]
            std_log_observed = np.std(log_observed_Rs, ddof=1)  # Use N-1 for sample std
            
            if std_log_observed > 0:  # Only do t-test if there's variation
                t_stat, p_value = ttest_1samp(log_observed_Rs, log_theoretical_R)
                effect_size = (log_theoretical_R - log_observed_R) / std_log_observed
            else:
                t_stat, p_value = 0.0, 1.0
                effect_size = 0.0
        else:
            log_observed_R = float('inf') if not observed_Rs else np.log(observed_Rs[0])
            t_stat, p_value = 0.0, 1.0
            effect_size = 0.0
        
        return {
            "log_theoretical_R": log_theoretical_R,
            "log_observed_R": log_observed_R,
            "p_value": p_value,
            "effect_size": effect_size
        }

class ModelValidator(RamseyValidator):
    """Extended validator for testing real language models."""
    
    def extract_embeddings(self, model: nn.Module, 
                          tokens: torch.Tensor) -> np.ndarray:
        """Extract embeddings from a given layer of the model."""
        with torch.no_grad():
            embeddings = model.embed_tokens(tokens)
        return embeddings.cpu().numpy()
    
    def compute_subspace_coherence(self, embeddings: np.ndarray, 
                                 subspace: np.ndarray) -> float:
        """Compute how well token differences align with a given subspace."""
        # Project differences onto subspace
        diffs = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                diffs.append(embeddings[i] - embeddings[j])
        diffs = np.stack(diffs)
        
        # Compute projection error
        proj = np.dot(diffs, subspace.T)
        reconstructed = np.dot(proj, subspace)
        errors = np.linalg.norm(diffs - reconstructed, axis=1)
        
        return 1.0 - np.mean(errors)
    
    def compare_hypotheses(self, model: nn.Module, 
                          tokens: torch.Tensor) -> Dict[str, float]:
        """Phase 2: Compare GRH vs LRH predictions."""
        embeddings = self.extract_embeddings(model, tokens)
        
        # GRH: Find Ramsey-theoretic subspaces
        grh_clique, grh_error = self.find_monochromatic_clique(embeddings)
        
        # LRH: Use PCA to find linear subspaces
        pca = PCA(n_components=self.config.k)
        pca.fit(embeddings)
        lrh_subspace = pca.components_
        
        # Compare coherence scores
        grh_coherence = self.compute_subspace_coherence(embeddings, grh_clique)
        lrh_coherence = self.compute_subspace_coherence(embeddings, lrh_subspace)
        
        return {
            "grh_coherence": grh_coherence,
            "lrh_coherence": lrh_coherence,
            "delta": grh_coherence - lrh_coherence,
            "p_value": ttest_1samp([grh_coherence - lrh_coherence], 0.0)[1]
        }
    
    def interpretability_analysis(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Phase 3: Evaluate practical utility of Ramsey subspaces."""
        # Find stable subspaces
        clique, _ = self.find_monochromatic_clique(embeddings)
        
        # Measure subspace stability under resampling
        stabilities = []
        for _ in range(self.config.num_trials):
            subset = embeddings[np.random.choice(len(embeddings), 
                                               size=len(embeddings)//2, 
                                               replace=False)]
            new_clique, _ = self.find_monochromatic_clique(subset)
            
            # Compute subspace overlap
            overlap = np.abs(np.dot(clique.T, new_clique)).mean()
            stabilities.append(overlap)
            
        return {
            "mean_stability": np.mean(stabilities),
            "std_stability": np.std(stabilities)
        }


@dataclass
class ModelConfig:
    name: str
    embedding_dim: int
    model_path: str

class ModelRegistry:
    """Registry of supported models and their configurations."""
    
    SUPPORTED_MODELS = {
        "gpt3": ModelConfig(
            name="gpt3",
            embedding_dim=12288,  # 12k dimensions
            model_path="EleutherAI/gpt-neox-20b"  # Using NeoX as GPT-3 proxy since actual GPT-3 isn't directly available
        ),
        "llama2": ModelConfig(
            name="llama2",
            embedding_dim=4096,  # 4k dimensions
            model_path="meta-llama/Llama-2-7b-hf"  # Using 7B variant
        )
    }
    
    @classmethod
    def get_config(cls, model_name: str) -> ModelConfig:
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. " 
                           f"Supported models: {list(cls.SUPPORTED_MODELS.keys())}")
        return cls.SUPPORTED_MODELS[model_name]

class ModelLoader:
    """Handles loading and preparation of different model architectures."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.logger = logging.getLogger("ModelLoader")
    
    def load_model_and_tokenizer(self, model_name: str) -> Tuple[nn.Module, AutoTokenizer]:
        """Load specified model and its tokenizer."""
        config = ModelRegistry.get_config(model_name)
        self.logger.info(f"Loading {model_name} from {config.model_path}")
        
        if model_name == "gpt3":
            # Using GPT-NeoX as a proxy for GPT-3 architecture
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(config.model_path)
            model = GPTNeoXModel.from_pretrained(config.model_path)
        elif model_name == "llama2":
            tokenizer = LlamaTokenizer.from_pretrained(config.model_path)
            model = LlamaModel.from_pretrained(config.model_path)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        return model, tokenizer

class EnhancedModelValidator(ModelValidator):
    """Extended validator with specific support for GPT-3 and LLaMA-2."""
    
    def __init__(self, config: ValidationConfig, model_name: str):
        super().__init__(config)
        self.model_name = model_name
        self.model_config = ModelRegistry.get_config(model_name)
        self.loader = ModelLoader()
    
    def extract_embeddings(self, model: nn.Module, 
                          tokens: torch.Tensor) -> np.ndarray:
        """Extract embeddings specific to each model architecture."""
        with torch.no_grad():
            if self.model_name == "gpt3":
                embeddings = model.embed_in(tokens)
            elif self.model_name == "llama2":
                embeddings = model.embed_tokens(tokens)
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
                
        return embeddings.cpu().numpy()
    
    def prepare_validation_data(self, tokenizer: AutoTokenizer, 
                              num_tokens: int = 1000) -> torch.Tensor:
        """Prepare a diverse set of tokens for validation."""
        # Generate a range of tokens that will produce diverse embeddings
        vocab_size = tokenizer.vocab_size
        token_ids = torch.randint(0, vocab_size, (num_tokens,))
        return token_ids.to(self.loader.device)
    
    def extract_model_embeddings(self, model: nn.Module, 
                               tokens: torch.Tensor) -> np.ndarray:
        """Extract embeddings specific to each model architecture."""
        with torch.no_grad():
            if self.model_name == "gpt3":
                embeddings = model.embed_in(tokens)
            elif self.model_name == "llama2":
                embeddings = model.embed_tokens(tokens)
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
                
        return embeddings.cpu().numpy()
    
    def compare_hypotheses(self, model: nn.Module, 
                          tokens: torch.Tensor) -> Dict[str, float]:
        """Phase 2: Compare GRH vs LRH predictions."""
        embeddings = self.extract_embeddings(model, tokens)
        
        # GRH: Find Ramsey-theoretic subspaces
        grh_clique, grh_error = self.find_monochromatic_clique(embeddings)
        
        # LRH: Use PCA to find linear subspaces
        pca = PCA(n_components=self.config.k)
        pca.fit(embeddings)
        lrh_subspace = pca.components_
        
        # Compare coherence scores
        grh_coherence = self.compute_subspace_coherence(embeddings, grh_clique)
        lrh_coherence = self.compute_subspace_coherence(embeddings, lrh_subspace)
        
        return {
            "grh_coherence": grh_coherence,
            "lrh_coherence": lrh_coherence,
            "delta": grh_coherence - lrh_coherence,
            "p_value": ttest_1samp([grh_coherence - lrh_coherence], 0.0)[1]
        }
    
    def interpretability_analysis(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Phase 3: Evaluate practical utility of Ramsey subspaces."""
        # Find stable subspaces
        clique, _ = self.find_monochromatic_clique(embeddings)
        
        # Measure subspace stability under resampling
        stabilities = []
        for _ in range(self.config.num_trials):
            subset = embeddings[np.random.choice(len(embeddings), 
                                               size=len(embeddings)//2, 
                                               replace=False)]
            new_clique, _ = self.find_monochromatic_clique(subset)
            
            # Compute subspace overlap
            overlap = np.abs(np.dot(clique.T, new_clique)).mean()
            stabilities.append(overlap)
            
        return {
            "mean_stability": np.mean(stabilities),
            "std_stability": np.std(stabilities)
        }
    
    def run_full_validation(self, num_tokens: int = 1000) -> Dict[str, Dict]:
        """Run complete validation suite on specified model."""
        # Load model and tokenizer
        model, tokenizer = self.loader.load_model_and_tokenizer(self.model_name)
        
        # Prepare validation data
        tokens = self.prepare_validation_data(tokenizer, num_tokens)
        
        # Extract embeddings
        embeddings = self.extract_model_embeddings(model, tokens)
        
        # Run all validation phases
        results = {
            "model_name": self.model_name,
            "embedding_dim": self.model_config.embedding_dim,
            "synthetic": self.synthetic_verification(),
            "comparison": self.compare_hypotheses(model, tokens),
            "interpretability": self.interpretability_analysis(embeddings)
        }
        
        return results

def main():
    # Test configuration for both models
    models_to_test = ["llama2"]
    
    for model_name in models_to_test:
        config = ValidationConfig(
            dimension=ModelRegistry.get_config(model_name).embedding_dim,
            k=3,  # testing 3D concept subspaces as in paper
            epsilon=0.1,
            num_samples=10000
        )
        
        validator = EnhancedModelValidator(config, model_name)
        
        try:
            results = validator.run_full_validation()
            print(f"\nResults for {model_name}:")
            print("=" * 50)
            for phase, metrics in results.items():
                if phase not in ["model_name", "embedding_dim"]:
                    print(f"\n{phase.capitalize()} Phase Results:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value}")
        except Exception as e:
            print(f"Error validating {model_name}: {str(e)}")

if __name__ == "__main__":
    main()