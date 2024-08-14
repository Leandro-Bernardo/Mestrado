import pandas as pd
import os

path_pmf_model_inference = os.path.join(os.path.dirname(__file__), "..",  "evaluation", "Chloride", "Chloride_official_model.xlsx")
path_descriptor_model_inference = os.path.join(os.path.dirname(__file__), "..", "evaluation", "Chloride", "with_blank", "statistics", "resnet50(4_blocks)(run true-sweep-6).xlsx")

pmf_model_inference = pd.read_excel(path_pmf_model_inference).set_index("Imagem da Amostra")
descriptor_model_inference = pd.read_excel(path_descriptor_model_inference).set_index('Unnamed: 0')

for sample in pmf_model_inference.index:
    descriptor_model_inference.loc[sample, "estimated"] = pmf_model_inference.loc[sample, "Estimativa Offline"]

descriptor_model_inference.to_excel(path_descriptor_model_inference)
print(" ")
