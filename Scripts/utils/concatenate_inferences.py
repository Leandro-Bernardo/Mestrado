import pandas as pd
import os

path_pmf_model_inference = os.path.join(os.path.dirname(__file__), "..",  "evaluation", "Chloride", "Chloride_Reinjecao_model.xlsx")
path_descriptor_model_inference = os.path.join(os.path.dirname(__file__), "..", "evaluation", "Chloride", "with_blank", "resnet50(3_blocks)", "best_model_3blocks_resnet50_img_size_448", "test", "statistics", "resnet50(3_blocks)(image_size_448).xlsx")

pmf_model_inference = pd.read_excel(path_pmf_model_inference).set_index("Imagem da Amostra")
descriptor_model_inference = pd.read_excel(path_descriptor_model_inference).set_index('Sample')

for sample in pmf_model_inference.index:
    descriptor_model_inference.loc[sample, "new_estimated"] = pmf_model_inference.loc[sample, "Estimativa Offline"]

descriptor_model_inference.to_excel(path_descriptor_model_inference)
print("Done!")
