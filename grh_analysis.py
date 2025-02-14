import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
import math
@dataclass
class AnalysisConfig:
    dimension: int
    k: int = 3
    epsilon: float = 0.1
    ensemble_size: int = 50
    subsample_ratio: float = 0.8
    stability_trials: int = 100
    expansion_attempts: int = 100  # Controls search depth
    neighborhood_decay: float = 0.9  # From GCE paper [6]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class GRHAnalyzer:
    """Implements all analyses from the paper with GPU acceleration"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = self._configure_logging()
        
    def _configure_logging(self) -> logging.Logger:
        logger = logging.getLogger("GRHAnalyzer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def synthetic_analysis(self, d: int, n_samples: int = 10000) -> Dict:
        embeddings = torch.randn(n_samples, d, device=self.config.device)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        
        results = {}
        for k in range(2, 6):
            clique, subspace, error = self._find_ramsey_clique(embeddings, k=k)
            coverage = self._compute_coverage(embeddings[clique], k) if len(clique) >= 2 else 0.0
            log_observed = math.log(len(clique)) if clique else 0.0
            
            results[f'k={k}'] = {
                'log_observed': log_observed,
                'log_theoretical': self._theoretical_threshold(k, d),
                'coverage': coverage,
                'error': error.item() if torch.is_tensor(error) else error
            }
        return results



    def subspace_comparison(self, embeddings: torch.Tensor) -> Dict:
        # Changed from _ramsey_clique_finder to direct _find_ramsey_clique
        grh_clique, grh_subspace, error = self._find_ramsey_clique(embeddings, k=self.config.k)
        lrh_subspace = self._pca_subspace(embeddings)
        
        return {
            'grh_coherence': self._subspace_coherence(embeddings, grh_subspace),
            'lrh_coherence': self._subspace_coherence(embeddings, lrh_subspace),
            'stability': self._subspace_stability(embeddings),
            'angle': self._subspace_angle(grh_subspace, lrh_subspace)
        }


    def concept_erasure(self, embeddings: torch.Tensor, 
                       eval_fn: callable) -> Dict:
        """Implements concept erasure protocol from Section 7.5"""
        grh_subspace = self.ensemble_grh(embeddings)
        erased = embeddings - embeddings @ grh_subspace @ grh_subspace.T
        
        return {
            'original_score': eval_fn(embeddings),
            'erased_score': eval_fn(erased),
            'delta': eval_fn(embeddings) - eval_fn(erased)
        }

    def dimensional_scaling(self, model, tokenizer, 
                           scales: List[float] = [1.0, 2.0, 4.0]) -> Dict:
        """Implements scaling laws analysis from Section 7.6"""
        results = {}
        base_emb = self._extract_base_embeddings(model, tokenizer)
        
        for scale in scales:
            scaled_emb = base_emb * scale + torch.randn_like(base_emb) * 0.01
            metrics = self.subspace_comparison(scaled_emb)
            results[f'scale_{scale}x'] = metrics
            
        return results

    def ensemble_grh(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Optimized implementation of Algorithm 1 with GPU acceleration"""
        M = self.config.ensemble_size
        ρ = self.config.subsample_ratio
        k = self.config.k
        
        subspaces = []
        for _ in range(M):
            sub_emb = self._stratified_subsample(embeddings, ρ)
            _, V = self._ramsey_clique_finder(sub_emb)
            subspaces.append(V)
            
        return self._grassmann_mean(subspaces)

    # Implementation Details --------------------------------------------------
    
    def _find_ramsey_clique(self, embeddings: torch.Tensor, k: int) -> Tuple:
        """Optimized Ramsey clique finder with dimensional safeguards"""
        n, d = embeddings.shape
        device = self.config.device
        
        # Initialize with proper subspace dimensions
        default_subspace = torch.zeros((d, k), device=device)
        best_clique = []
        best_subspace = default_subspace.clone()
        best_error = torch.tensor(float('inf'), device=device)
        
        # Precompute normalized embeddings for cosine similarity
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        normalized = embeddings / torch.clamp(norms, min=1e-7)
        
        for _ in range(1000):
            # Greedy expansion with vectorized operations
            candidates = self._greedy_clique_expansion(normalized, k)
            if len(candidates) < k + 1:
                continue
                
            # Compute pairwise differences
            clique_emb = embeddings[candidates]
            diffs = clique_emb.unsqueeze(1) - clique_emb.unsqueeze(0)
            diffs = diffs.view(-1, d)
            
            # Dimensional validation
            if diffs.size(0) < k or diffs.size(1) != d:
                continue
                
            # Regularized PCA
            try:
                _, _, V = torch.pca_lowrank(diffs, q=k, center=True)
                current_subspace = V[:, :k]
            except RuntimeError:
                continue
                
            # Projection and error calculation
            proj = diffs @ current_subspace @ current_subspace.T
            error = torch.max(torch.norm(diffs - proj, dim=1))
            
            # Update best found
            if error < best_error:
                best_error = error.clone()
                best_clique = candidates.copy()
                best_subspace = current_subspace.clone()
                
                # Early exit if ε-approximation achieved
                if best_error <= self.config.epsilon:
                    break
                    
        # Final validation
        if best_subspace.shape != (d, k):
            best_subspace = default_subspace.clone()
            
        return best_clique, best_subspace, best_error


    def _stratified_subsample(self, embeddings: torch.Tensor, ρ: float) -> torch.Tensor:
        """GPU-accelerated stratified sampling"""
        cluster_ids = MiniBatchKMeans(n_clusters=self.config.k).fit_predict(embeddings.cpu())
        cluster_ids = torch.tensor(cluster_ids, device=self.config.device)
        
        sampled = []
        for c in torch.unique(cluster_ids):
            mask = (cluster_ids == c)
            n_sample = int(ρ * mask.sum().item())
            sampled.append(torch.randperm(mask.sum())[:n_sample])
            
        return embeddings[torch.cat(sampled)]

    def _grassmann_mean(self, subspaces: List[torch.Tensor]) -> torch.Tensor:
        """Stable Grassmann mean calculation with regularization"""
        U = torch.stack(subspaces)
        UUT = torch.bmm(U, U.transpose(1,2))
        mean_UUT = UUT.mean(dim=0) + 1e-6*torch.eye(U.shape[-1], device=U.device)
        _, _, V = torch.svd(mean_UUT)
        return V[:, :self.config.k]

    # Helper Methods ----------------------------------------------------------
    
    def _subspace_coherence(self, embeddings: torch.Tensor, subspace: torch.Tensor) -> float:
        """Robust subspace coherence calculation"""
        if subspace.numel() == 1 or subspace.ndim != 2:
            return 0.0
        
        d, k = embeddings.shape[1], self.config.k
        if subspace.shape != (d, k):
            subspace = torch.zeros((d, k), device=subspace.device)
        
        proj = embeddings @ subspace @ subspace.T
        errors = torch.norm(embeddings - proj, dim=1)
        return 1 - errors.mean().item()



    def _subspace_stability(self, embeddings: torch.Tensor) -> float:
        half = embeddings.shape[0] // 2
        S1 = self.ensemble_grh(embeddings[:half])
        S2 = self.ensemble_grh(embeddings[half:])
        angles = subspace_angles(S1.cpu().numpy(), S2.cpu().numpy())
        return np.cos(angles).mean()

    def _theoretical_threshold(self, k: int, d: int) -> float:
        """Logarithmic implementation to prevent overflow"""
        c = math.e  # Empirical constant from paper
        eps = self.config.epsilon
        
        log_term = k**2 * (d - k) * (math.log(c) - math.log(eps))
        return math.log(k + 1) + log_term

    
    def _greedy_clique_expansion(self, embeddings: torch.Tensor, k: int) -> List[int]:
        """GPU-accelerated greedy clique expansion with seed diversification"""
        n = embeddings.shape[0]
        best_clique = []
        max_attempts = 100  # From search result [2] PMC4603673
        
        for _ in range(max_attempts):
            # Random seed selection with geometric diversification
            seed = torch.randint(0, n, (1,)).item()
            current_clique = [seed]
            
            # Compute initial adjacency mask
            adj_mask = torch.all(embeddings[seed] - embeddings < self.config.epsilon, dim=1)
            
            for _ in range(k-1):
                valid_nodes = torch.where(adj_mask)[0]
                if len(valid_nodes) == 0:
                    break
                
                # Select node with maximum common neighbors (GCE algorithm [6])
                # neighbor_counts = torch.sum(adj_mask[valid_nodes], dim=1)
                neighbor_counts = torch.sum(adj_mask[valid_nodes].float(), dim=0)  # Remove invalid dimension
                next_node = valid_nodes[torch.argmax(neighbor_counts)]
                
                current_clique.append(next_node.item())
                adj_mask &= torch.all(embeddings[next_node] - embeddings < self.config.epsilon, dim=1)

            # Update best clique found
            if len(current_clique) > len(best_clique):
                best_clique = current_clique
                if len(best_clique) >= k+1:  # Early exit if target size reached
                    break

        return best_clique[:k+1]  # Return top k+1 candidates
    
    def _clique_approximation_error(self, clique_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute maximum subspace approximation error for a clique"""
        if len(clique_embeddings) < 2:
            return torch.tensor(float('inf'), device=self.config.device)
        
        # Get dimension from embeddings instead of config
        dim = clique_embeddings.shape[1]
        
        # Compute all pairwise differences
        diffs = clique_embeddings.unsqueeze(1) - clique_embeddings.unsqueeze(0)
        diffs = diffs.view(-1, dim)  # Use dynamic dimension
        
        if len(clique_embeddings) < 2:
            return torch.tensor(float('inf'), device=self.config.device)
        
        # Compute all pairwise differences
        diffs = clique_embeddings.unsqueeze(1) - clique_embeddings.unsqueeze(0)
        diffs = diffs.view(-1, self.config.dimension)  # Flatten to N*(N-1)/2 x d
        
        # Handle edge case with insufficient differences
        if diffs.size(0) < self.config.k:
            return torch.tensor(float('inf'), device=self.config.device)
        
        # Compute PCA subspace
        _, _, V = torch.pca_lowrank(diffs, q=self.config.k)
        subspace = V[:, :self.config.k]
        
        # Project differences and compute residuals
        proj = diffs @ subspace @ subspace.T
        residuals = diffs - proj
        errors = torch.norm(residuals, dim=1)
        
        return torch.max(errors)
    
    def _compute_coverage(self, clique_embeddings: torch.Tensor, k: int) -> float:
        """Computes subspace coverage via explained variance ratio"""
        n = clique_embeddings.size(0)
        if n < 2:
            return 0.0
        
        # Compute all pairwise differences
        diffs = clique_embeddings.unsqueeze(1) - clique_embeddings.unsqueeze(0)
        diffs = diffs.view(-1, self.config.dimension)
        
        # Handle edge case with insufficient differences
        if diffs.size(0) < k:
            return 0.0
        
        # Perform PCA on differences
        _, S, V = torch.pca_lowrank(diffs, q=k, center=False)
        
        # Calculate explained variance ratio
        total_variance = torch.sum(diffs.pow(2)) / (diffs.size(0) - 1)
        explained_variance = torch.sum(S[:k]**2) / (diffs.size(0) - 1)
        
        return (explained_variance / total_variance).item()
    
    def _ramsey_clique_finder(self, embeddings: torch.Tensor) -> Tuple:
        """Wrapper for existing functionality"""
        return self._find_ramsey_clique(embeddings, k=self.config.k)

    def _pca_subspace(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute PCA subspace using PyTorch SVD with centering"""
        # Center the embeddings
        mean = torch.mean(embeddings, dim=0)
        centered = embeddings - mean
        
        # Compute SVD
        _, _, V = torch.pca_lowrank(centered, q=self.config.k)
        return V[:, :self.config.k]

    def _subspace_angle(self, A: torch.Tensor, B: torch.Tensor) -> float:
        """Compute principal angles between subspaces using SciPy"""
        from scipy.linalg import subspace_angles
        
        # Convert to numpy arrays and ensure column vectors
        A_np = A.cpu().numpy().T if A.shape[1] == self.config.k else A.cpu().numpy()
        B_np = B.cpu().numpy().T if B.shape[1] == self.config.k else B.cpu().numpy()
        
        angles = subspace_angles(A_np, B_np)
        return np.mean(np.cos(angles))






# Example Usage ---------------------------------------------------------------

if __name__ == "__main__":
    config = AnalysisConfig(dimension=4096, k=3, epsilon=0.1)
    analyzer = GRHAnalyzer(config)
    
    # Synthetic Analysis
    synthetic_results = analyzer.synthetic_analysis(d=4096)
    print("Synthetic Results:", synthetic_results)
    
    # Model Analysis (Example with Dummy Data)
    dummy_embeddings = torch.randn(1000, 4096, device=config.device)
    comparison_results = analyzer.subspace_comparison(dummy_embeddings)
    print("\nSubspace Comparison:", comparison_results)
