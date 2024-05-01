import os
import cv2
import torch
import torchvision

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import vgg11
#from torchsummary import summary
from torchvision.models.feature_extraction import create_feature_extractor

# Variables
ANALYTE = "Chloride"
SKIP_BLANK = False

if SKIP_BLANK:
    os.makedirs(os.path.join("..", "descriptors", f"{ANALYTE}", "no_blank"), exist_ok = True)
    LOAD_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank"))
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "descriptors", f"{ANALYTE}", "no_blank")
else:
    os.makedirs(os.path.join("..", "descriptors", f"{ANALYTE}", "with_blank"), exist_ok = True)
    LOAD_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank"))
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "descriptors", f"{ANALYTE}", "with_blank")

# load images
cropped_images = []

for i in range(int(len(os.listdir(LOAD_PATH))/3)):
        cropped_images.append(cv2.imread(os.path.join(LOAD_PATH, f"sample_{i}.png")))

#print(summary(vgg11(), (3, 224, 224)))

#cria o symbolic trace do torch.FX
train_nodes, eval_nodes = get_graph_node_names(vgg11())
#print(train_nodes)

feature_list = ['features.2', 'features.5', 'features.10']

#cria o extrator de caracteristicas
feature_extraction = create_feature_extractor(
                                              model = vgg11(),
                                              return_nodes= feature_list
                                              )

# extract the features

image_tensor = {}

for i, _ in enumerate(cropped_images):
    img_choice = cv2.cvtColor(cropped_images[i].astype('uint8'),cv2.COLOR_BGR2RGB) # escolhe uma imagem

    img_choice = torch.from_numpy(img_choice).permute(2, 0, 1).unsqueeze(0).float()  # converte a imagem para o formato de tensor

    with torch.no_grad():
        image_tensor[f"sample_{i}"] = feature_extraction(img_choice)  # dada imagem, extrai as caracteristicas de todas camadas da vgg


# extract the features from all images

for sample_key, sample in image_tensor.items():  #extrai caracteristicas das camadas de convolucao desejadas (primeira, segunda e terceira)
    print(f"Amostra atual : {sample_key}")
    features2 = sample['features.2'][0]
    features5 = sample['features.5'][0]
    features10 = sample['features.10'][0]

    heigh, width = features2.shape[1], features2.shape[2]

    #reescala os mapas de caracteristicas da segunda e da terceira camada para o tamanho da primeira camada
    features5_rescaled = torchvision.transforms.Resize((heigh, width),torchvision.transforms.InterpolationMode.NEAREST)(features5)    #reescala output da segunda camada
    features10_rescaled = torchvision.transforms.Resize((heigh, width), torchvision.transforms.InterpolationMode.NEAREST)(features10) #reescala output da terceira camada

    #concatena todas as imagens em um unico tensor e faz o cropp baseado no campo receptivo da terceira camada (22 x 22)
    sample_features = torch.cat((features2, features5_rescaled, features10_rescaled), dim=0)                       # shape: 448 , 112, 112
    sample_features = sample_features[:, 9 : sample_features.shape[1] - 9,  9 : sample_features.shape[2] - 9]  # shape: 448 ,  90,  90
    sample_features = torch.flatten(torch.permute(sample_features, (1, 2, 0)), start_dim=0, end_dim=1)  # flatten    shape: num_vectors, num_channels

    #le o valor de alcalinidade (dado) e expande a dimensão (atualmente 1d) para o tamanho do descritor da sua respectiva imagem
    sample_theoretical_value = torch.tensor(float(open(os.path.join(LOAD_PATH, f"{sample_key}.txt")).read())).expand(len(sample_features))

    torch.save(sample_features, os.path.join(SAVE_PATH, f"{sample_key}"))
    torch.save(sample_theoretical_value, os.path.join(SAVE_PATH, f"{sample_key}_anotation"))


