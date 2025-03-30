import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Generator, Discriminator  
import os
from PIL import Image
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import matplotlib.pyplot as plt
from torchvision import transforms as T
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01, feat = None):

        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device       
        self.feat = feat
        self.rand = True
    def perturb(self, X_nat, y, c_trg):

        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_() 

        for i in range(self.k):
            X.requires_grad = True
            output= self.model(X, c_trg)
            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad
            X_adv = X + self.a * grad.sign()
            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()
        self.model.zero_grad()
        return X, X - X_nat

class Solver(object):
    def __init__(self, config):
        """Initialize configurations."""
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.dataset = config.dataset
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.selected_attrs = config.selected_attrs
        self.test_iters = config.test_iters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = config.model_save_dir
        self.build_model()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_model(self):
        """Create a generator and a discriminator.""" 
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.G.to(self.device)
        self.D.to(self.device)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator.""" 
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing.""" 
      
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices: 
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list
    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):

        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.clone()
        
            if i < c_org.size(1): 
                if dataset == 'CelebA':
                    if i in hair_color_indices:  
                        c_trg[:, i] = 1
                        for j in hair_color_indices:
                            if j != i:
                                c_trg[:, j] = 0
                    else:
                        c_trg[:, i] = (c_trg[:, i] == 0)  
                elif dataset == 'RaFD':
                    c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            
                c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

 
def model1_stargan(imagepath, processedpath):
    args = SimpleNamespace()
    args.dataset = "CelebA"
    args.selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
    args.g_lr = 0.0001
    args.d_lr = 0.0001
    args.beta1 = 0.5
    args.beta2 = 0.999
    args.test_iters = 200000
    args.c_dim = 5
    args.c2_dim = 8
    args.image_size = 256
    args.g_conv_dim = 64
    args.d_conv_dim = 64
    args.g_repeat_num = 6
    args.d_repeat_num = 6
    args.model_save_dir = r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\stargan\stargan_celeba_256\models"
    args.num_workers = 1
    args.mode = "test"
    print(args)
    solver = Solver(args)
    solver.restore_model(solver.test_iters) 
    transform = T.Compose([
        T.Resize(256),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(imagepath)
    image = transform(image).unsqueeze(0).to(device)
    c_org = torch.zeros((1, solver.c_dim)).to(device)  
    attack = LinfPGDAttack(model=solver.G, device=device, epsilon=0.1, k=20, a=0.03)
    c_trg = c_org.clone()  
    X_adv,delta= attack.perturb(image, image, c_trg)
    X_adv = X_adv.squeeze().cpu().detach().numpy().transpose(1, 2, 0)  
    X_adv = (X_adv + 1) / 2
    plt.imsave(processedpath, X_adv)
    print(f"Adversarial image saved at {processedpath}")
    
model1_stargan(
    imagepath=r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\young_man.jpg",
    processedpath=r"C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\combine_with_gradio\model1_adv_img.jpg"
)