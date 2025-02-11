import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict, Optional, Union
from transformers import (
    GPTNeoXModel, 
    LlamaModel, 
    GPTNeoXTokenizer, 
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from dataclasses import dataclass
import logging
from tqdm import tqdm

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
            model_path="meta-llama/Llama-2-7b"  # Using 7B variant
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
            tokenizer = GPTNeoXTokenizer.from_pretrained(config.model_path)
            model = GPTNeoXModel.from_pretrained(config.model_path)
        elif model_name == "llama2":
            tokenizer = LlamaTokenizer.from_pretrained(config.model_path)
            model = LlamaModel.from_pretrained(config.model_path)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        return model, tokenizer

class EnhancedModelValidator(RamseyValidator):
    """Extended validator with specific support for GPT-3 and LLaMA-2."""
    
    def __init__(self, config: ValidationConfig, model_name: str):
        super().__init__(config)
        self.model_name = model_name
        self.model_config = ModelRegistry.get_config(model_name)
        self.loader = ModelLoader()
        
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
    models_to_test = ["gpt3", "llama2"]
    
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