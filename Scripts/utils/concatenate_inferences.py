import pandas as pd
import os

ANALYTE = 'Chloride'
LOSS = 'mse'
IMG_TO_EVALUATE = 'test'

path_pmf_model_inference = os.path.join(os.path.dirname(__file__), "..",  "evaluation", f"{ANALYTE}", "planilhas_da_dissertacao", LOSS, f"{ANALYTE}_reinjecao_{IMG_TO_EVALUATE}_samples.xlsx")
path_descriptor_model_inference = os.path.join(os.path.dirname(__file__), "..", "evaluation", f"{ANALYTE}", "planilhas_da_dissertacao", LOSS,f"{ANALYTE}_{IMG_TO_EVALUATE}_samples.xlsx")

pmf_model_inference = pd.read_excel(path_pmf_model_inference).set_index("Imagem da Amostra")
try:
    descriptor_model_inference = pd.read_excel(path_descriptor_model_inference).set_index('Unnamed: 0') #.set_index('Sample')
    descriptor_model_inference.index.name = 'Sample'
except:
    descriptor_model_inference = pd.read_excel(path_descriptor_model_inference).set_index('Sample')

for sample in pmf_model_inference.index:
    descriptor_model_inference.loc[sample, "estimated"] = pmf_model_inference.loc[sample, "Estimativa Offline"]

descriptor_model_inference.to_excel(path_descriptor_model_inference)
print("Done!")
