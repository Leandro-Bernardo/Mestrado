import os
import cv2
import torch
import torchvision

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import vgg11
from torchsummary import summary
from torchvision.models.feature_extraction import create_feature_extractor

os.makedirs("../descriptors", exist_ok =True)

# load images
cropped_images = []

for i in range(int(len(os.listdir("../images/"))/2)):
        cropped_images.append(cv2.imread(f"../images/sample_{i}.png"))

print(summary(vgg11(), (3, 224, 224)))

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

    img_choice = torch.from_numpy(img_choice).permute(2, 0, 1).unsqueeze(0).float()  #converte a imagem para o formato de tensor

    with torch.no_grad():
        image_tensor[f"sample_{i}"] = feature_extraction(img_choice)


# extract the features from all images

for sample_key, sample in image_tensor.items():
    print(f"Amostra atual : {sample_key}")
    features2 = sample['features.2'][0]
    features5 = sample['features.5'][0]
    features10 = sample['features.10'][0]

    heigh, width = features2.shape[1], features2.shape[2]

    #reescala os mapas de caracteristicas da segunda e da terceira camada para o tamanho da primeira camada
    #features5_rescaled, features10_rescaled  = cv2.resize(features5, (-1, heigh, width), cv2.INTER_NEAREST), cv2.resize(features10, (-1, heigh, width), cv2.INTER_NEAREST)
    features5_rescaled = torchvision.transforms.Resize((heigh, width),torchvision.transforms.InterpolationMode.NEAREST)(features5)
    features10_rescaled = torchvision.transforms.Resize((heigh, width), torchvision.transforms.InterpolationMode.NEAREST)(features10)

    sample_features = torch.cat((features2, features5_rescaled, features10_rescaled), dim=0)
    sample_features = torch.flatten(torch.permute(sample_features, (1, 2, 0)), start_dim=0, end_dim=1)  # shape=(num_vectors, num_channels)
    
    #sample_theoretical_value = theoretical_value.expand(len(sample_features))  # shape = (num_vectors,)
    sample_theoretical_value = torch.tensor(float(open(f"../images/{sample_key}.txt").read())).expand(len(sample_features)) 

    torch.save(sample_features, f"../descriptors/{sample_key}")
    torch.save(sample_theoretical_value,f"../descriptors/{sample_key}_anotation")
    
# image_tensor = {}

# for i, _ in enumerate(cropped_images):
#     img_choice = cv2.cvtColor(cropped_images[i].astype('uint8'),cv2.COLOR_BGR2RGB) # escolhe uma imagem

#     img_choice = torch.from_numpy(img_choice).permute(2, 0, 1).unsqueeze(0).float()  #converte a imagem para o formato de tensor

#     with torch.no_grad():
#         image_tensor[f"sample_{i}"] = feature_extraction(img_choice)

