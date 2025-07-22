import pandas as pd
import os
import numpy as np

ANALYTE = 'Chloride'
LOSS = 'mse'


path = os.path.join(os.path.dirname(__file__), "..", "evaluation", f"{ANALYTE}", "planilhas_da_dissertacao", LOSS)
excel_path = os.path.join(path, f"{ANALYTE}_planilha_dissertacao_sem_inf.xlsx")

try:
    excel = pd.read_excel(excel_path).set_index('Sample')
    excel.index.name = 'Sample'
except:
    excel = pd.read_excel(excel_path)

stats = ['variance', 'std', 'mad', 'relative error (mean)', 'relative error (median)', 'relative error (mode)']
avg_values = {metric: excel[metric].mean() for metric in stats}
avg_df = pd.DataFrame([avg_values])

avg_df.to_excel(os.path.join(path,f'avg_statistics.xlsx'))