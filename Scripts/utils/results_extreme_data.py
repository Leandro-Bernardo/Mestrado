import pandas as pd

import os

ANALYTE = 'Chloride'
LOSS = 'mse'
METRIC = 'median'
num_of_samples = 3
path = os.path.join(os.path.dirname(__file__), "..", "evaluation", f"{ANALYTE}", "planilhas_da_dissertacao", LOSS)
excel_path = os.path.join(path, f"{ANALYTE}_planilha_dissertacao_sem_inf.xlsx")

try:
    excel = pd.read_excel(excel_path).set_index('Sample')
    excel.index.name = 'Sample'
except:
    excel = pd.read_excel(excel_path)

min = excel.sort_values(by=f'relative error ({METRIC})', ascending=True)
max = excel.sort_values(by=f'relative error ({METRIC})', ascending=False)#.sort_values(by='relative error (median)', ascending=True)

values = pd.concat([min.head(num_of_samples), max.head(num_of_samples)])
values.to_excel(os.path.join(path,f'{num_of_samples}_min_max_samples({METRIC}).xlsx'))