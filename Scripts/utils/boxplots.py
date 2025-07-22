import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from scipy import stats
from matplotlib.lines import Line2D

ANALYTE = 'Chloride'
LOSS = 'emd'
num_of_samples = 40
CREATE_BOXPLOTS = True
MODEL_NAME = 'volcanic' + '-sweep-' + '54'

with open(os.path.join(".", "settings.yaml"), "r") as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    FEATURE_EXTRACTOR = settings["feature_extractor"]
    CNN1_OUTPUT_SHAPE =  settings["feature_extraction"][FEATURE_EXTRACTOR][ANALYTE]["cnn1_output_shape"]

#read data
path = os.path.join(os.path.dirname(__file__), "..", "evaluation", f"{ANALYTE}", "planilhas_da_dissertacao", LOSS,)
val_samples=  pd.read_csv(os.path.join(path, f"{MODEL_NAME}(val).txt"))
test_samples = pd.read_csv(os.path.join(path, f"{MODEL_NAME}(test).txt"))

#concat test and val values
cat = pd.concat([val_samples, test_samples], axis=0, ignore_index=True)

#split predicted value and expected value into two numpy array
predicted_value = cat.loc[:, "predicted_value"].to_numpy()
expected_value = cat.loc[:, "expected_value"].to_numpy()

#gets min and max values
min_value = np.min(predicted_value) - 0.1*np.min(predicted_value)
if ANALYTE == 'Ph':
     max_value = 10.90
else:
    max_value =  np.max(predicted_value) + 0.1*np.max(predicted_value)

#reshapes data into samples
predicted_value_for_samples = {f"amostra_{i+1}" : value for i, value in enumerate(np.reshape(predicted_value,( -1, CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE)))}
expected_value_from_samples = {f"amostra_{i+1}" : value for i, value in enumerate(np.reshape(expected_value,( -1, CNN1_OUTPUT_SHAPE * CNN1_OUTPUT_SHAPE)))}

#reads the excel file with predicted value from PMF model
excel_path = os.path.join(path, f"{ANALYTE}_planilha_dissertacao.xlsx")
excel = pd.read_excel(excel_path).fillna(0.0)
# try:
#     excel = pd.read_excel(excel_path).set_index('Sample').fillna(0.0)
#     excel.index.name = 'Sample'
# except:
#     excel = pd.read_excel(excel_path).fillna(0.0)


#verify invalid values (not presented in excel) and fixes it
valid_predicted_values_from_samples = {}
valid_expected_values_from_samples = {}
valid_estimated_value_from_samples = {}
excel_rows = []
excel_row_names = []
count = 1
excel_id = 0
for i in range(1, len(expected_value_from_samples.keys())):
    try:
        if round(expected_value_from_samples[f'amostra_{i}'][0], 2) == round(excel.loc[excel_id, 'expected value'], 2):
            valid_predicted_values_from_samples[f'amostra_{count}'] = predicted_value_for_samples[f'amostra_{i}']
            valid_expected_values_from_samples[f'amostra_{count}'] = expected_value_from_samples[f'amostra_{i}']
            valid_estimated_value_from_samples[f'amostra_{count}'] = round(excel.loc[excel_id, 'estimated'], 2)
            excel_rows.append(excel.loc[excel_id, :])
            excel_row_names.append(f'amostra_{count}')
            count+=1
            excel_id+=1
        else:
            print(f'invalid sample: {i} ', '  expected value (from sample):', round(expected_value_from_samples[f'amostra_{i}'][0], 2) , '  expected value (from excel):', round(excel.loc[excel_id, 'expected value'], 2))
    except:
        pass

valid_excel_lines = pd.DataFrame(excel_rows, index=excel_row_names)

chosen_samples = sample(list(valid_expected_values_from_samples.keys()), k=num_of_samples)
chosen_samples_excel = valid_excel_lines.loc[chosen_samples].to_excel(f"{path}\\{MODEL_NAME}_chosen_samples(boxplot).xlsx")

data = [] #pd.DataFrame()
for sample in chosen_samples:
    for val in predicted_value_for_samples[sample]:
        data.append({'Samples': sample, 'Prediction': val})

df = pd.DataFrame(data)

ordered_samples = chosen_samples

# Horizontal
# plt.figure(figsize=(22, 10))
# ax = sns.boxplot(x='Samples', y='Prediction', data=df, order=ordered_samples, boxprops=dict(alpha=0.6))


# # Adiciona as linhas esperadas e estatísticas
# xticks = ax.get_xticks()  # posições reais no eixo x
# for i, sample in enumerate(ordered_samples):
#     values = df[df['Samples'] == sample]['Prediction'].values
#     expected = valid_expected_values_from_samples[sample][0]

#     mean = np.mean(values)
#     median = np.median(values)
#     mode = stats.mode(values, keepdims=True)[0][0]

#     xpos = xticks[i]  # posição correta no eixo x

#     ax.hlines(expected, xpos - 0.3, xpos + 0.3, colors='gray', linestyles='dashed', label='Esperado' if i == 0 else "")
#     ax.hlines(mean,     xpos - 0.3, xpos + 0.3, colors='red', label='Média' if i == 0 else "")
#     ax.hlines(median,   xpos - 0.3, xpos + 0.3, colors='blue', label='Mediana' if i == 0 else "")
#     ax.hlines(mode,     xpos - 0.3, xpos + 0.3, colors='green', label='Moda' if i == 0 else "")

# Vertical
plt.figure(figsize=(20, 22))

flierprops = dict(marker='o', markerfacecolor='gray', markersize=5, alpha=0.3)
boxprops = dict(facecolor='none', edgecolor='black', linewidth=1)
medianprops = dict(visible=False)
ax = sns.boxplot(y='Samples', x='Prediction', data=df, order=ordered_samples, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, orient='h')

ax.set_xlim(min_value, max_value)

plt.draw()
# adds an extra X axis on top
ax_top = ax.twiny()
ax_top.set_xlim(min_value, max_value)
xticks = np.linspace(min_value, max_value, num=6)

ax_top.set_xticks(xticks)
ax_top.set_xticklabels([f"{tick:.2f}" for tick in xticks])

# plt.draw()

# for label in ax_top.get_xticklabels():
#     label.set_rotation(0)  # horizontal
#     label.set_verticalalignment('bottom')

ax_top.set_xlabel("")
ax.set_xlabel("")
ax_top.grid(False)

yticks = ax.get_yticks()

for i, sample in enumerate(ordered_samples):
    values = df[df['Samples'] == sample]['Prediction'].values
    expected = valid_expected_values_from_samples[sample][0]

    mean = np.mean(values)
    median = np.median(values)
    mode = stats.mode(values, keepdims=True)[0][0]
    estimated = valid_estimated_value_from_samples[sample]

    ypos = yticks[i]

    ax.vlines(expected, ypos - 0.3, ypos + 0.3, colors='gray', linestyles='dashdot', label='Expected' if i == 0 else "")
    ax.vlines(estimated, ypos - 0.3, ypos + 0.3, colors='#FFA500', linestyles='dashdot', label='Estimated' if i == 0 else "")
    ax.vlines(mean,     ypos - 0.3, ypos + 0.3, colors='#FF0000', label='Mean' if i == 0 else "")
    ax.vlines(median,   ypos - 0.3, ypos + 0.3, colors='#0000FF', label='Median' if i == 0 else "")
    ax.vlines(mode,     ypos - 0.3, ypos + 0.3, colors='#00FF00', label='Mode' if i == 0 else "")

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5)
    legend_elements = [
    Line2D([0], [0], color='gray', linestyle='dashdot', label='Esperado'),
    Line2D([0], [0], color='orange', linestyle='dashdot', label='Estimado'),
    Line2D([0], [0], color='red', label='Media'),
    Line2D([0], [0], color='blue', label='Mediana'),
    Line2D([0], [0], color='green', label='Moda'),

]


ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=5)
#plt.title("Boxplot ")
#plt.xlabel("Predictions")
plt.ylabel("Samples")
plt.xticks(rotation=90)
plt.grid(True)
plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.draw()

for label in ax_top.get_xticklabels():
    label.set_rotation(0)  # horizontal
    label.set_verticalalignment('bottom')

plt.tight_layout(rect=[0, 0, 0.85, 0.97])
#plt.tight_layout()

pdf_path = os.path.join(path, f"{MODEL_NAME}_boxplot_with_stats.pdf")
png_path = os.path.join(path, f"{MODEL_NAME}_boxplot_with_stats.png")
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
