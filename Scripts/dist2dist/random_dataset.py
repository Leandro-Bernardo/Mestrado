import numpy as np
from torch.utils.data import Dataset, TensorDataset
from chemical_analysis.typing import BoundedAnalyte
from typing import List, Tuple

class ItemwiseDataset(Dataset):
    
    def __init__(self, bounded_analyte: List[BoundedAnalyte], subset: TensorDataset, afine_parameters: Tuple[np.ndarray, np.ndarray, float, np.ndarray], dims: int):
        self.bounded_analyte = bounded_analyte
        self.subset = subset
        self.afine_parameters = afine_parameters
        self.dims = dims
    
    def __len__(self,):
        return len(self.bounded_analyte)
    
    def __getitem__(self, idx):
        bounded: BoundedAnalyte = self.bounded_analyte[idx]
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