from torch.utils.data import Dataset, TensorDataset
from typing import List
from chemical_analysis.typing import BoundedAnalyte


class GeneratedDataset(Dataset):
    def __init__(self, bounded_analyte: List[BoundedAnalyte], subset: TensorDataset):
        self.bounded_analyte = bounded_analyte
        self.subset = subset
    
    def __len__(self):
        return len(self.bounded_analyte)
    
    def __getitem__(self, idx):
        bounded: BoundedAnalyte = self.bounded_analyte[idx]
        value = bounded['target']['value']
        lower_pmf = self.subset[bounded['bounders']['lower']['index']][0]
        upper_pmf = self.subset[bounded['bounders']['upper']['index']][0]
        inter_factor = bounded['interpolation_factor']
        
        return value, lower_pmf, upper_pmf, inter_factor