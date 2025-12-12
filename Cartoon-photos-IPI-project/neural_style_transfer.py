import cv2
import os
import shutil
import numpy as np
import scipy.ndimage.filters

import torch
import torch.optim as optim
import torch.nn.functional as F
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
    global cnt_features
    _loss = 0
    for i in range(len(yhat)):
        _loss += F.mse_loss(cnt_features[i], yhat[i])
    
    return _loss/len(yhat)

def gram(x):
    b, c, h, w = x.size()
    x = x.view(b*c, -1)
    return torch.mm(x, x.t())

def style_loss(yhat):
    global sty_features
    _loss = 0
    for i in range(len(yhat)):
        style_gram = gram(sty_features[i])
        ip_gram = gram(yhat[i])
        _loss += F.mse_loss(style_gram, ip_gram)

    return _loss/len(yhat)

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
    global cnt_sfs
    global sty_sfs
    global cnt_features
    global sty_features
    global ip
    global optimizer
    global i

    gpu0 = torch.device("cuda:0")

    if os.path.exists('debug'):
        shutil.rmtree('debug')
    os.makedirs('debug')

    orig_np, style_np = get_imgs(orig_path, style_path)

    # CORREÇÃO 1: Definir a normalização e aplicá-la corretamente
    # Note que a normalização espera tensor (C, H, W)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    model = getattr(models, MODEL)
    vgg = model(pretrained=True)
    
    # Prepara tensores, envia para GPU e normaliza
    cnt = torch.tensor(orig_np).float().to(gpu0)
    cnt = normalize(cnt)
    
    style = torch.tensor(style_np).float().to(gpu0)
    style = normalize(style)

    # pegando apenas as camadas convolucionais do VGG-16
    vgg = torch.nn.Sequential(*list(vgg.features.children())[:43]).to(gpu0)
    
    # CORREÇÃO 2: Travar o modelo em modo de avaliação (CRUCIAL para VGG_BN)
    vgg.eval() 
    
    for param in vgg.parameters():
        param.requires_grad = False

    layers = [5, 12, 22]
    cnt_sfs = [SaveFeatures(vgg[i]) for i in layers]

    # passa a imagem e pega as features desejadas
    # Adiciona dimensão do batch com unsqueeze(0)
    vgg(cnt.unsqueeze(0))
    
    # CORREÇÃO 3: Detach para garantir que o alvo seja constante (boa prática)
    cnt_features = [sf.features.clone().detach() for sf in cnt_sfs]

    layers = [5, 12, 22, 32, 42]
    sty_sfs = [SaveFeatures(vgg[i]) for i in layers]
    vgg(style.unsqueeze(0))
    sty_features = [sf.features.clone().detach() for sf in sty_sfs]

    # Imagem inicial com ruído
    np_ip = np.random.uniform(0.0, 1.0, size=orig_np.shape)
    np_ip = scipy.ndimage.filters.median_filter(np_ip, [8,8,1])
    
    # Normaliza a imagem de input inicial também
    ip_tensor = torch.tensor(np_ip).float().to(gpu0)
    ip_tensor = normalize(ip_tensor)
    
    # Habilita gradiente na imagem de entrada
    ip = ip_tensor.unsqueeze(0).requires_grad_(True)

    optimizer = optim.LBFGS([ip], lr=1) # LR pode precisar de ajuste dependendo da escala

    i = 0
    
    # Função auxiliar para desnormalizar e salvar imagem (para visualização correta)
    def save_debug_image(tensor_img, step_count):
        # Desfaz normalização para salvar
        t_img = tensor_img.clone().detach().cpu().squeeze()
        
        # Desnormalização manual: x * std + mean
        for t, m, s in zip(t_img, norm_mean, norm_std):
            t.mul_(s).add_(m)
            
        out_img = t_img.permute(1, 2, 0).numpy()
        out_img = np.clip(out_img, 0, 1)
        cv2.imwrite(os.path.join('debug', str(step_count) + '.png'), out_img * 255)

    # Atualizando a função step para usar o novo salvamento
    def step_closure():
        global i
        c_w = 1e9  # Pesos geralmente precisam ser ajustados
        s_w = 1e5 # Peso de estilo costuma ser muito maior que conteúdo
        
        optimizer.zero_grad()
        vgg(ip)
        
        cnt_ip_features = [sf.features.clone() for sf in cnt_sfs]
        sty_ip_features = [sf.features.clone() for sf in sty_sfs]
        
        cnt_loss = c_w * content_loss(cnt_ip_features)
        sty_loss = s_w * style_loss(sty_ip_features)
        loss = cnt_loss + sty_loss
        loss.backward()

        if i % 100 == 0:
            print(f"Step - {i} Content: {cnt_loss.item():.4f}, Style: {sty_loss.item():.4f}, Total: {loss.item():.4f}")
            save_debug_image(ip, i)
            
        i += 1
        return loss

    while i < STEPS:
        optimizer.step(step_closure)

neural_style_transfer("golfinho.jpg", "steamboat_willie.jpg")