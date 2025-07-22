import pandas as pd
import numpy as np
from random import choices
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml


ANALYTE = 'Chloride'
LOSS = 'mse'
num_of_samples = 45

path = os.path.join(os.path.dirname(__file__), "..", "evaluation", f"{ANALYTE}", "planilhas_da_dissertacao", LOSS)
excel_path = os.path.join(path, f"{ANALYTE}_planilha_dissertacao.xlsx")

try:
    excel = pd.read_excel(excel_path).set_index('Sample')
    excel.index.name = 'Sample'
except:
    excel = pd.read_excel(excel_path)

chosen_samples = choices(excel.index, k = num_of_samples)

chosen_samples_df = pd.DataFrame()
for sample in chosen_samples:
    chosen_samples_df[sample] = excel.loc[sample,:]

chosen_samples_df = chosen_samples_df.transpose()
chosen_samples_df.to_excel(os.path.join(path,'chosen_samples.xlsx'))