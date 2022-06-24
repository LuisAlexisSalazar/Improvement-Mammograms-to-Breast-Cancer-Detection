from .VGG16 import create_VGG16
from .VGG19 import create_VGG19
from .Resnet50 import create_ResNet50
from .MobilNet2 import create_mobilNet
from .CNN import create_basic_cnn_model

# *Si usaremos alguna de estasMobilNet VGG16 VGG19 ..
transfer_learning = True


def is_transfer_learning(name_model):
    global transfer_learning
    if name_model is ["CNN", "MobilNet2"]:
        transfer_learning = False


# Nueva red EfficientNetB3
# https://www.kaggle.com/code/fanconic/efficientnetb3-train-keras/notebook

def create_model(name_model, mode_classification):
    is_transfer_learning(name_model)
    if name_model == "CNN":
        return create_basic_cnn_model(mode_classification)
    elif name_model == "MobilNet2":
        return create_mobilNet(mode_classification)
    elif name_model == "VGG16":
        return create_VGG16(mode_classification)
    elif name_model == "VGG19":
        return create_VGG19(mode_classification)
    elif name_model == "ResNet50":
        return create_ResNet50(mode_classification)
    else:
        print("Nombre de Red valida")
    return None
# return create_mobilNet(mode_classification)
