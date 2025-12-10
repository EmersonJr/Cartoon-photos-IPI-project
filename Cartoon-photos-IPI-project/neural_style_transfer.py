import cv2
import os
import shutil
import numpy as np
import scipy.ndimage.filters

import torch
import torch.optim as optim
import torch.nn.functional as functional
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

MODEL = 'vgg16_bn'
STEPS = 10000
MAX_H = 400
MAX_W = 400

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
    loss = 0
    for i in range(len(yhat)):
        loss += functional.mse_loss(orig_features[i], yhat[i])
    
    return loss/len(yhat)

def gram(x):
    b, c, h, w = x.size()
    x = x.view(b*c, -1)
    return torch.mm(x, x.t())

def style_loss(yhat):
    loss = 0
    for i in range(len(yhat)):
        style_gram = gram(style_features[i])
        ip_gram = gram(yhat[i])
        loss += functional.mse_loss(style_gram, ip_gram)

    return loss/len(yhat)

def step():
    global optimizer

    c_w = 0.001
    s_w = 0.001
    vgg(ip)
    orig_ip_features = [sf.features.clone() for sf in orig_sfs]
    style_ip_features = [sf.features.clone() for sf in style_sfs]
    orig_loss = c_w * content_loss(orig_ip_features)
    sty_loss = s_w * style_loss(style_ip_features)
    loss = orig_loss + sty_loss
    optimizer.zero_grad()
    loss.backward()

    if i % 100:
        print("Step - {} Content loss - {}, Style loss - {}, Total loss - {}".format(
            i, orig_loss.data[0], sty_loss.data[0], loss.data[0]))
        out_img = ip.data.cpu().squeeze().permute(1, 2, 0).numpy()
        cv2.imwrite(os.path.join('debug', str(i) + '.png'), out_img*255);

    return loss

def read_img(path):
    img = cv2.imread(path)
    if img.shape[0] > MAX_H and img.shape[1] > MAX_W:
        img = cv2.resize(img, (MAX_W, MAX_H))
    
    return img / 255.0

def get_imgs(path_orig, path_style):
    
    orig = read_img(path_orig)
    style = read_img(path_style)

    orig = np.transpose(orig, (2,0,1))
    style = np.transpose(style, (2,0,1))

    return orig, style

def neural_style_transfer(orig_path, style_path):
    global vgg
    global orig_sfs
    global style_sfs
    global orig_features
    global style_features
    global ip
    global optimizer
    global i

    gpu0 = torch.device("cuda:0")

    if os.path.exists('debug'):
        shutil.rmtree('debug')
    os.makedirs('debug')

    orig_np, style_np = get_imgs(orig_path, style_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = getattr(models, MODEL)
    vgg = model(pretrained=True)
    orig, style = torch.Tensor(orig_np), torch.Tensor(style_np)

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

    # imagem inicial eh uma aleatoria, com ruido, para treinarmos e igualarmos
    np_ip = np.random.uniform(0.0, 1.0, size=orig_np.shape)
    np_ip = scipy.ndimage.filters.median_filter(np_ip, [8,8,1])
    ip = torch.Tensor(np_ip)[None].to(gpu0)
    ip = Variable(ip, requires_grad=True)

    optimizer = optim.LBFGS([ip], lr=0.005)

    i = 0
    while i < STEPS:
        optimizer.step(step)
        i += 1