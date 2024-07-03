import os
import cv2
import torch
import torchvision
import numpy as np
import json

from torch import FloatTensor, UntypedStorage
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import vgg11
#from torchsummary import summary
from torchvision.models.feature_extraction import create_feature_extractor

# Variables
ANALYTE = "Alkalinity"
SKIP_BLANK = True

if SKIP_BLANK:
    LOAD_TRAIN_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank", "train"))
    LOAD_VAL_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank", "val"))
    LOAD_TEST_PATH = (os.path.join("..", "images", f"{ANALYTE}", "no_blank", "test"))
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "no_blank")
else:
    LOAD_TRAIN_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank", "train"))
    LOAD_VAL_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank", "val"))
    LOAD_TEST_PATH = (os.path.join("..", "images", f"{ANALYTE}", "with_blank", "test"))
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "Udescriptors", f"{ANALYTE}", "with_blank")

if ANALYTE == "Alkalinity":
    feature_list = ['features.2', 'features.5', 'features.10']
    RECEPTIVE_FIELD_DIM = 15
    DESCRIPTOR_DEPTH = 448
    IMAGE_SIZE = 98 * 98  # after the crop based on the receptive field  (shape = (112 - 15, 112 - 15))

elif ANALYTE == "Chloride":
    feature_list = ['features.2', 'features.5', 'features.10', 'features.15', 'features.20']
    RECEPTIVE_FIELD_DIM = 27
    DESCRIPTOR_DEPTH = 1472
    IMAGE_SIZE = 86 * 86  # after the crop based on the receptive field  (shape = (112 - 27, 112 - 27))

TOTAL_TRAIN_SAMPLES = (int(len(os.listdir(LOAD_TRAIN_PATH))/3))
TOTAL_VAL_SAMPLES = (int(len(os.listdir(LOAD_VAL_PATH))/3))
TOTAL_TEST_SAMPLES = (int(len(os.listdir(LOAD_TEST_PATH))/3))

os.makedirs(SAVE_PATH, exist_ok = True)


def main(load_path,  total_samples, stage):
    #loads images
    cropped_images = []

    for i in range(int(len(os.listdir(load_path))/3)):
            cropped_images.append(cv2.imread(os.path.join(load_path, f"sample_{i}.png")))

    #print(summary(vgg11(), (3, 224, 224)))

    #cria o symbolic trace do torch.FX
    train_nodes, eval_nodes = get_graph_node_names(vgg11())
    #print(train_nodes)

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


    # creat untyped storage object for save the descriptors
    nbytes_float32 = torch.finfo(torch.float32).bits//8
    descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(SAVE_PATH, f"descriptors_{stage}.bin"), shared = True, nbytes= (total_samples * IMAGE_SIZE * DESCRIPTOR_DEPTH) * nbytes_float32)).view(total_samples, IMAGE_SIZE, DESCRIPTOR_DEPTH)
    expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(SAVE_PATH, f"descriptors_anotation_{stage}.bin"), shared = True, nbytes= (total_samples * IMAGE_SIZE) * nbytes_float32)).view(total_samples, IMAGE_SIZE)
    # storage_index = 0
    # dim = total_samples * IMAGE_SIZE
    # nbytes_float32 = torch.finfo(torch.float32).bits//8
    # descriptors = FloatTensor(UntypedStorage.from_file(os.path.join(SAVE_PATH, f"descriptors_{stage}.bin"), shared = True, nbytes= (dim * DESCRIPTOR_DEPTH) * nbytes_float32)).view(dim, DESCRIPTOR_DEPTH)
    # expected_value = FloatTensor(UntypedStorage.from_file(os.path.join(SAVE_PATH, f"descriptors_anotation_{stage}.bin"), shared = True, nbytes= (dim) * nbytes_float32))#.view(dim)

    # extract the features from all images
    if ANALYTE == "Alkalinity":
        for i, (sample_key, sample) in enumerate(image_tensor.items()):  #extrai caracteristicas das camadas de convolucao desejadas (primeira, segunda e terceira)
            print(f"Amostra atual : {sample_key},  Total: {total_samples}")
            features2 = sample['features.2'][0]
            features5 = sample['features.5'][0]
            features10 = sample['features.10'][0]

            heigh, width = features2.shape[1], features2.shape[2]

            #reescala os mapas de caracteristicas da segunda e da terceira camada para o tamanho da primeira camada
            features5_rescaled = torchvision.transforms.Resize((heigh, width),torchvision.transforms.InterpolationMode.NEAREST)(features5).to(torch.float32)    #reescala output da segunda camada
            features10_rescaled = torchvision.transforms.Resize((heigh, width), torchvision.transforms.InterpolationMode.NEAREST)(features10).to(torch.float32) #reescala output da terceira camada

            #concatena todas as imagens em um unico tensor e faz o cropp baseado no campo receptivo da terceira camada (15 x 15)
            sample_features = torch.cat((features2, features5_rescaled, features10_rescaled,), dim=0)  # shape: 1472 , 112, 112
            rf = int((RECEPTIVE_FIELD_DIM - 1)/2)  # (27-1)/2  =  13
            sample_features = sample_features[:, rf : sample_features.shape[1] - rf,  rf : sample_features.shape[2] - rf]  # shape: 448 ,  97,  97
            sample_features = torch.permute(sample_features, (1, 2, 0))   # shape:  97,  97,  448
            sample_features = torch.reshape(sample_features, (-1, DESCRIPTOR_DEPTH))  # shape:  9604 (image size),  448
            #le o valor do analito e expande a dimensão (atualmente 1d) para o tamanho do descritor da sua respectiva imagem
            sample_theoretical_value = torch.tensor(float(open(os.path.join(load_path, f"{sample_key}.txt")).read())).expand(len(sample_features)).to(torch.float32)

            assert sample_features.shape[0] == IMAGE_SIZE

            descriptors[i, ...] = sample_features
            expected_value[i, ...] = sample_theoretical_value

        # for sample_key, sample in image_tensor.items():  #extrai caracteristicas das camadas de convolucao desejadas (primeira, segunda e terceira)
        #     print(f"Amostra atual : {sample_key},  Total: {total_samples} ")
        #     features2 = sample['features.2'][0]
        #     features5 = sample['features.5'][0]
        #     features10 = sample['features.10'][0]

        #     heigh, width = features2.shape[1], features2.shape[2]

        #     #reescala os mapas de caracteristicas da segunda e da terceira camada para o tamanho da primeira camada
        #     features5_rescaled = torchvision.transforms.Resize((heigh, width),torchvision.transforms.InterpolationMode.NEAREST)(features5)    #reescala output da segunda camada
        #     features10_rescaled = torchvision.transforms.Resize((heigh, width), torchvision.transforms.InterpolationMode.NEAREST)(features10) #reescala output da terceira camada

        #     #concatena todas as imagens em um unico tensor e faz o cropp baseado no campo receptivo da
        #     sample_features = torch.cat((features2, features5_rescaled, features10_rescaled), dim=0)                       # shape: 448 , 112, 112
        #     rf = int((RECEPTIVE_FIELD_DIM - 1)/2)  # (15-1)/2  =  7
        #     sample_features = sample_features[:, rf : sample_features.shape[1] - rf,  rf : sample_features.shape[2] - rf] # shape: 448 ,  105,  105
        #     sample_features = torch.flatten(torch.permute(sample_features, (1, 2, 0)), start_dim=0, end_dim=1)  # flatten    shape: num_vectors, num_channels

        #     #le o valor do analito e expande a dimensão (atualmente 1d) para o tamanho do descritor da sua respectiva imagem
        #     sample_theoretical_value = torch.tensor(float(open(os.path.join(load_path, f"{sample_key}.txt")).read())).expand(len(sample_features))

        #     # torch.save(sample_features, os.path.join(SAVE_PATH, f"{sample_key}"))
        #     # torch.save(sample_theoretical_value, os.path.join(SAVE_PATH, f"{sample_key}_anotation"))

        #     for index, _ in enumerate(sample_features):
        #         assert sample_features.shape[0] == IMAGE_SIZE

        #         descriptors[storage_index, ...] = sample_features[index]
        #         expected_value[storage_index] = sample_theoretical_value[index]

        #         storage_index +=1


    elif ANALYTE == "Chloride":
        for i, (sample_key, sample) in enumerate(image_tensor.items()):  #extrai caracteristicas das camadas de convolucao desejadas (primeira, segunda e terceira)
            print(f"Amostra atual : {sample_key},  Total: {total_samples}")
            features2 = sample['features.2'][0]
            features5 = sample['features.5'][0]
            features10 = sample['features.10'][0]
            features15 = sample['features.15'][0]
            features20 = sample['features.20'][0]

            heigh, width = features2.shape[1], features2.shape[2]

            #reescala os mapas de caracteristicas da segunda e da terceira camada para o tamanho da primeira camada
            features5_rescaled = torchvision.transforms.Resize((heigh, width),torchvision.transforms.InterpolationMode.NEAREST)(features5).to(torch.float32)    #reescala output da segunda camada
            features10_rescaled = torchvision.transforms.Resize((heigh, width), torchvision.transforms.InterpolationMode.NEAREST)(features10).to(torch.float32) #reescala output da terceira camada
            features15_rescaled = torchvision.transforms.Resize((heigh, width), torchvision.transforms.InterpolationMode.NEAREST)(features15).to(torch.float32) #reescala output da quarta camada
            features20_rescaled = torchvision.transforms.Resize((heigh, width), torchvision.transforms.InterpolationMode.NEAREST)(features20).to(torch.float32) #reescala output da quinta camada

            #concatena todas as imagens em um unico tensor e faz o cropp baseado no campo receptivo da quinta camada (27 x 27)
            sample_features = torch.cat((features2, features5_rescaled, features10_rescaled, features15_rescaled, features20_rescaled), dim=0)  # shape: 1472 , 112, 112
            rf = int((RECEPTIVE_FIELD_DIM - 1)/2)  # (27-1)/2  =  13
            sample_features = sample_features[:, rf : sample_features.shape[1] - rf,  rf : sample_features.shape[2] - rf]  # shape: 1472 ,  86,  86
            sample_features = torch.permute(sample_features, (1, 2, 0))   # shape:  86,  86,  1472
            sample_features = torch.reshape(sample_features, (-1, DESCRIPTOR_DEPTH))  # shape:  7396 (image size),  1472
            #le o valor do analito e expande a dimensão (atualmente 1d) para o tamanho do descritor da sua respectiva imagem
            sample_theoretical_value = torch.tensor(float(open(os.path.join(load_path, f"{sample_key}.txt")).read())).expand(len(sample_features)).to(torch.float32)

            assert sample_features.shape[0] == IMAGE_SIZE

            descriptors[i, ...] = sample_features
            expected_value[i, ...] = sample_theoretical_value



    with open(os.path.join(SAVE_PATH, f"metadata_{stage}.json"), "w") as file:
        json.dump({
            "total_samples": total_samples,
            "image_size": IMAGE_SIZE,
            "descriptor_depth": DESCRIPTOR_DEPTH
        }, file)

if __name__ == "__main__":
    main(LOAD_TRAIN_PATH, TOTAL_TRAIN_SAMPLES, "train")
    main(LOAD_VAL_PATH, TOTAL_VAL_SAMPLES, "val")
    main(LOAD_TEST_PATH, TOTAL_TEST_SAMPLES, "test")