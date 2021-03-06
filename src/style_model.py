import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Loadung the model vgg19 that will serve as the base model
model = models.vgg19(pretrained=True).features

# Assigning the GPU to the variable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ["0", "5", "10", "19", "28"]
        # Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model = models.vgg19(pretrained=True).features[
            :29
        ]  # model will contain the first 29 layers

    # x holds the input tensor(image) that will be feeded to each layer
    def forward(self, x):
        # initialize an array that wil hold the activations from the chosen layers
        features = []
        # Iterate over all the layers of the mode
        for layer_num, layer in enumerate(self.model):
            # activation of the layer will stored in x
            x = layer(x)
            # appending the activation of the selected layers and return the feature array
            if str(layer_num) in self.req_features:
                features.append(x)
        return features


# defing a function that will load the image and perform the required preprocessing and put it on the GPU
def image_loader(path):
    image = Image.open(path).convert("RGB")
    # defining the image transformation steps to be performed before feeding them to the model
    loader = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    # The preprocessing steps involves resizing the image and then converting it to a tensor

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def calc_content_loss(gen_feat, orig_feat):
    # calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l


def calc_style_loss(gen, style):
    # Calculating the gram matrix for the style and the generated image
    batch_size, channel, height, width = gen.shape
    G = torch.mm(
        gen.view(channel, height * width), gen.view(channel, height * width).t()
    )
    A = torch.mm(
        style.view(channel, height * width), style.view(channel, height * width).t()
    )
    # Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l = torch.mean((G - A) ** 2)  # /(4*channel*(height*width)**2)
    return style_l


def calc_style_loss(gen, style):
    # Calculating the gram matrix for the style and the generated image
    batch_size, channel, height, width = gen.shape

    G = torch.mm(
        gen.view(channel, height * width), gen.view(channel, height * width).t()
    )
    A = torch.mm(
        style.view(channel, height * width), style.view(channel, height * width).t()
    )

    # Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l = torch.mean((G - A) ** 2)
    return style_l


def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss = content_loss = 0
    for gen, cont, style in zip(gen_features, orig_feautes, style_featues):
        # extracting the dimensions from the generated image
        content_loss += calc_content_loss(gen, cont)
        style_loss += calc_style_loss(gen, style)

    # calculating the total loss of e th epoch
    total_loss = 8 * content_loss + 70 * style_loss
    return total_loss


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find("Linear") != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
