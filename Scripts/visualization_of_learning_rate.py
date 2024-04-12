import matplotlib.pyplot as plt
import pandas as pd
import os

MODEL_VERSION = "model_1"

data = pd.read_csv(os.path.join(os.path.dirname(__file__),"learning_values",f"{MODEL_VERSION}.txt"), sep="\t")
print(data)

plt.plot(data["Epoch"][10:], data["Loss"][10:])
plt.show()