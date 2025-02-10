import numpy as np
import random
from torch.utils.data import Dataset, TensorDataset
from chemical_analysis.typing import BoundedAnalyte, GroupedBoundedAnalyte
from typing import List, Tuple


class CyclicDataset(Dataset):
    def __init__(self, bounded_analyte: List[BoundedAnalyte], subset: TensorDataset, afine_parameters: Tuple[np.ndarray, np.ndarray, float, np.ndarray], grouped_bounded_analyte: List[GroupedBoundedAnalyte], dims: int):
        self.bounded_analyte = bounded_analyte
        self.subset = subset
        self.afine_parameters = afine_parameters
        self.grouped_bounded_analyte = grouped_bounded_analyte
        self.dims = dims
        self.count = np.zeros(len(self.grouped_bounded_analyte), dtype=int)
        self.candidate_sizes = self.get_cand_size()
    
    def __len__(self) -> int:
        return len(self.grouped_bounded_analyte)
    
    def get_cand_size(self):
        sizes = [len(d['indexes']) for d in self.grouped_bounded_analyte]
        return np.array(sizes)
    
    def select_candidate(self, candidates: List[int], idx: int) -> int:
        chosen_idx = self.count[idx] % self.candidate_sizes[idx]
        self.count[idx] += 1
        return int(candidates[chosen_idx])
    
    def __getitem__(self, idx):
        item: GroupedBoundedAnalyte = self.grouped_bounded_analyte[idx]
        candidates_idx = item['indexes']
        chosen_idx = self.select_candidate(candidates=candidates_idx, idx=idx)
        
        bounded: BoundedAnalyte = self.bounded_analyte[chosen_idx]
        values = bounded['target']['value']
        target_pmfs = self.subset[bounded['target']['index']][0]
        lower_pmfs = self.subset[bounded['bounders']['lower']['index']][0]
        upper_pmfs = self.subset[bounded['bounders']['upper']['index']][0]
        inter_factors = bounded['interpolation_factor']
        
        if self.dims > 1:
            trans_centers = self.afine_parameters[bounded['target']['index']][0]
            trans_translations = self.afine_parameters[bounded['target']['index']][1]
            trans_angles = self.afine_parameters[bounded['target']['index']][2]
            trans_target_pmfs = self.afine_parameters[bounded['target']['index']][3]
        else:
            trans_centers = 0.0
            trans_translations = 0.0
            trans_angles = 0.0
            trans_target_pmfs = 0.0
        
        return values, target_pmfs, lower_pmfs, upper_pmfs, inter_factors, trans_centers, trans_translations, trans_angles, trans_target_pmfs