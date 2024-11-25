from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as utl
import random
import torch
import hydra
import math
import os
import pickle
from torchvision.utils import save_image
class CustomData:
    def __init__(self, data_dir, dataset_name, number_data_points, mix=None, only_mix=True, noise=0):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.data_points = number_data_points
        self.mix = mix
        self.om = only_mix
        self.noise = noise
        self.extract_mean_std()

    def get_data_cfg(self):
        with hydra.initialize(config_path='../breaching/config/case/data', version_base='1.1'):
            cfg = hydra.compose(config_name=self.dataset_name)
        return cfg

    def extract_mean_std(self):
        cfg = self.get_data_cfg()
        self.mean = torch.as_tensor(cfg.mean)[None,:,None,None]
        self.std  = torch.as_tensor(cfg.std)[None,:,None,None]

    def process_data(self, sec_input4=False):
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224,224)),
            ]
        )
        # trans = transforms.ToTensor()
        file_name_li = os.listdir(self.data_dir)
        file_name_list = sorted(file_name_li, key=lambda x: int(x.split('-')[0]))
        assert len(file_name_list) >= self.data_points
        imgs = []
        labels_ = []
        # random.seed(219)
        # random.shuffle(file_name_list)
        #####################################
        for file_name in file_name_list[0:self.data_points]:
            img = Image.open(self.data_dir+file_name).convert("RGB")
            imgs.append(trans(img)[None,:])
            label = int(file_name.split('-')[0])
            # label = 20
            labels_.append(label)
        imgs = torch.cat(imgs, 0)
        labels = torch.tensor(labels_)
        inputs = (imgs-self.mean)/self.std
        alpha  = 1-self.noise
        inputs = pow(alpha, 0.5)*inputs + pow(1-alpha, 0.5)*torch.randn_like(inputs)
        if self.mix:
            pass
        if sec_input4:
            pass
        return dict(inputs=inputs, labels=labels)

    def get_initial_from_img(self, path):
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
            ]
        )
        img = trans(Image.open(path))[None,:]
        return (img-self.mean)/self.std

    # def save_recover(self, recover, original=None, save_pth='', sature=False):
    #     using_sqrt_row = False
    #     if original is not None:
    #         if isinstance(recover, dict):
    #             batch = recover['data'].shape[0]
    #             recover_imgs = torch.clamp((recover['data'].cpu()*self.std+self.mean), 0, 1)
    #             if sature:
    #                 recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
    #             origina_imgs = torch.clamp((original['data'].cpu()*self.std+self.mean), 0, 1)
    #             all = torch.cat([recover_imgs, origina_imgs], 0)
    #             if using_sqrt_row:
    #                 utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
    #             else:
    #                 utl.save_image(all, save_pth, nrow=batch)
    #         else:
    #             batch = recover.shape[0]
    #             recover_imgs = torch.clamp((recover.cpu()*self.std+self.mean), 0, 1)
    #             if sature:
    #                 recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
    #             origina_imgs = torch.clamp((original['data'].cpu()*self.std+self.mean), 0, 1)
    #             all = torch.cat([recover_imgs, origina_imgs], 0)
    #             if using_sqrt_row:
    #                 utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
    #             else:
    #                 utl.save_image(all, save_pth, nrow=batch)
    #     else:
    #         if isinstance(recover, dict):
    #             batch = recover['data'].shape[0]
    #             recover_imgs = torch.clamp((recover['data'].cpu()*self.std+self.mean), 0, 1)
    #             if sature:
    #                 recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
    #             all = recover_imgs
    #             if using_sqrt_row:
    #                 utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
    #             else:
    #                 utl.save_image(all, save_pth, nrow=batch)
    #         else:
    #             batch = recover.shape[0]
    #             recover_imgs = torch.clamp((recover.cpu()*self.std+self.mean), 0, 1)
    #             if sature:
    #                 recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
    #             all = recover_imgs
    #             if using_sqrt_row:
    #                 utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
    #             else:
    #                 utl.save_image(all, save_pth, nrow=batch)
    def save_recover(self, recover, original=None, save_pth='', sature=False):
        # 使用传入的save_pth的目录路径
        save_dir = os.path.dirname(save_pth)
        filename = os.path.splitext(os.path.basename(save_pth))[0]
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建单张图片保存的子文件夹
        single_img_dir = os.path.join(save_dir, f'{filename}_single_images')
        os.makedirs(single_img_dir, exist_ok=True)
        
        def save_raw_data(data, name):
            raw_path = os.path.join(save_dir, f'{filename}_{name}_raw.pkl')
            with open(raw_path, 'wb') as f:
                pickle.dump(data.cpu().detach(), f)
                
        def save_single_images(images, prefix):
            for i in range(images.shape[0]):
                img_path = os.path.join(single_img_dir, f'{prefix}_{i}.jpg')
                save_image(images[i], img_path)
        
        using_sqrt_row = False
        if original is not None:
            if isinstance(recover, dict):
                batch = recover['data'].shape[0]
                recover_imgs = torch.clamp((recover['data'].cpu()*self.std+self.mean), 0, 1)
                # 保存recover相关数据
                save_raw_data(recover['data'], 'recover')
                save_single_images(recover_imgs, 'recover')
                
                if sature:
                    recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
                origina_imgs = torch.clamp((original['data'].cpu()*self.std+self.mean), 0, 1)
                # 保存original相关数据
                save_raw_data(original['data'], 'original')
                save_single_images(origina_imgs, 'original')
                
                all = torch.cat([recover_imgs, origina_imgs], 0)
                if using_sqrt_row:
                    utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
                else:
                    utl.save_image(all, save_pth, nrow=batch)
            else:
                batch = recover.shape[0]
                recover_imgs = torch.clamp((recover.cpu()*self.std+self.mean), 0, 1)
                # 保存recover相关数据
                save_raw_data(recover, 'recover')
                save_single_images(recover_imgs, 'recover')
                
                if sature:
                    recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
                origina_imgs = torch.clamp((original['data'].cpu()*self.std+self.mean), 0, 1)
                # 保存original相关数据
                save_raw_data(original['data'], 'original')
                save_single_images(origina_imgs, 'original')
                
                all = torch.cat([recover_imgs, origina_imgs], 0)
                if using_sqrt_row:
                    utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
                else:
                    utl.save_image(all, save_pth, nrow=batch)
        else:
            if isinstance(recover, dict):
                batch = recover['data'].shape[0]
                recover_imgs = torch.clamp((recover['data'].cpu()*self.std+self.mean), 0, 1)
                # 保存recover相关数据
                save_raw_data(recover['data'], 'recover')
                save_single_images(recover_imgs, 'recover')
                
                if sature:
                    recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
                all = recover_imgs
                if using_sqrt_row:
                    utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
                else:
                    utl.save_image(all, save_pth, nrow=batch)
            else:
                batch = recover.shape[0]
                recover_imgs = torch.clamp((recover.cpu()*self.std+self.mean), 0, 1)
                # 保存recover相关数据
                save_raw_data(recover, 'recover')
                save_single_images(recover_imgs, 'recover')
                
                if sature:
                    recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
                all = recover_imgs
                if using_sqrt_row:
                    utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
                else:
                    utl.save_image(all, save_pth, nrow=batch)
    
    # def save_recover(self, recover, original=None, save_pth='', sature=False):
    #     import pickle
    #     import os
        
    #     # 获取不带后缀的文件路径
    #     save_dir = os.path.dirname(save_pth)
    #     filename = os.path.splitext(os.path.basename(save_pth))[0]
        
    #     using_sqrt_row = False
        
    #     def save_raw_data(data, name):
    #         # 保存原始tensor数据
    #         raw_path = os.path.join(save_dir, f'{filename}_{name}_raw.pkl')
    #         with open(raw_path, 'wb') as f:
    #             pickle.dump(data.cpu().detach(), f)
        
    #     if original is not None:
    #         if isinstance(recover, dict):
    #             batch = recover['data'].shape[0]
    #             recover_imgs = torch.clamp((recover['data'].cpu()*self.std+self.mean), 0, 1)
    #             # 保存recover的原始数据
    #             save_raw_data(recover['data'], 'recover')
                
    #             if sature:
    #                 recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
    #             origina_imgs = torch.clamp((original['data'].cpu()*self.std+self.mean), 0, 1)
    #             # 保存original的原始数据
    #             save_raw_data(original['data'], 'original')
                
    #             all = torch.cat([recover_imgs, origina_imgs], 0)
    #             if using_sqrt_row:
    #                 utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
    #             else:
    #                 utl.save_image(all, save_pth, nrow=batch)
    #         else:
    #             batch = recover.shape[0]
    #             recover_imgs = torch.clamp((recover.cpu()*self.std+self.mean), 0, 1)
    #             # 保存recover的原始数据
    #             save_raw_data(recover, 'recover')
                
    #             if sature:
    #                 recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
    #             origina_imgs = torch.clamp((original['data'].cpu()*self.std+self.mean), 0, 1)
    #             # 保存original的原始数据
    #             save_raw_data(original['data'], 'original')
                
    #             all = torch.cat([recover_imgs, origina_imgs], 0)
    #             if using_sqrt_row:
    #                 utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
    #             else:
    #                 utl.save_image(all, save_pth, nrow=batch)
    #     else:
    #         if isinstance(recover, dict):
    #             batch = recover['data'].shape[0]
    #             recover_imgs = torch.clamp((recover['data'].cpu()*self.std+self.mean), 0, 1)
    #             # 保存recover的原始数据
    #             save_raw_data(recover['data'], 'recover')
                
    #             if sature:
    #                 recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
    #             all = recover_imgs
    #             if using_sqrt_row:
    #                 utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
    #             else:
    #                 utl.save_image(all, save_pth, nrow=batch)
    #         else:
    #             batch = recover.shape[0]
    #             recover_imgs = torch.clamp((recover.cpu()*self.std+self.mean), 0, 1)
    #             # 保存recover的原始数据
    #             save_raw_data(recover, 'recover')
                
    #             if sature:
    #                 recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
    #             all = recover_imgs
    #             if using_sqrt_row:
    #                 utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
    #             else:
    #                 utl.save_image(all, save_pth, nrow=batch)

    def recover_to_0_1(self, recover):
        tmp = recover['data'].data.clone()
        trans = torch.clamp((tmp.cpu()*self.std+self.mean), 0, 1)
        return trans

    def pixel_0_1_to_norm(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0).clamp(0, 1)
        else: tensor = tensor.clamp(0, 1)
        return (tensor-self.mean)/self.std
