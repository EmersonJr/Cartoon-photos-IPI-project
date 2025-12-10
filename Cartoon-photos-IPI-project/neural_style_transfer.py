import torch
import torch.nn.functional as functional
import torchvision.models as models
from torch.autograd import Variable

MODEL = 'vgg16_bn'

# classe usada para salvar as ativacoes de camadas convolucionais
class SaveFeatures(torch.nn.Module):
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


def content_loss(yhat):
    global orig_features

    loss = 0
    for i in range(len(yhat)):
        loss += functional.mse_loss(orig_features[i], yhat[i])
    
    return loss/len(yhat)

def gram(x):
    b, c, h, w = x.size()
    x = x.view(b*c, -1)
    return torch.mm(x, x.t())

def style_loss(yhat):
    global style_features

    loss = 0
    for i in range(len(yhat)):
        style_gram = gram(style_features[i])
        ip_gram = gram(yhat[i])
        loss += functional.mse_loss(style_gram, ip_gram)

    return loss/len(yhat)

def get_imgs():
    # STUB por enquanto
    return 

def neural_style_transfer():
    global orig_features
    global style_features

    gpu0 = torch.device("cuda:0")

    orig, style = get_imgs()

    model = getattr(models, MODEL)
    vgg = model(pretrained=True)

    # pegando apenas as camadas convolucionais do VGG-16
    vgg = torch.nn.Sequential(*list(vgg.features.children())[:43]).cuda()
    for param in vgg.parameters():
        param.requires_grad = False

    layers = [5, 12, 22]
    orig_sfs = [SaveFeatures(vgg[i]) for i in layers]

    # passa a imagem e pega as features desejadas
    vgg(Variable(orig[None].to(gpu0)))
    orig_features = [sf.features.clone() for sf in orig_sfs]

    layers = [5, 12, 22, 32, 42]
    style_sfs = [SaveFeatures(vgg[i]) for i in layers]
    vgg(Variable(style[None].to(gpu0)))
    style_features = [sf.features.clone() for sf in style_sfs]

