import pandas as pd
from random import choices
import os

ANALYTE = 'Chloride'
num_of_samples = 40

path = os.path.join(os.path.dirname(__file__), "..", "evaluation", f"{ANALYTE}")
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