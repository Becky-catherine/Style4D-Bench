import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import net as net
from sampler import InfiniteSamplerWrapper

#训练集预处理
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)

        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

#调整学习率
def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

#创建训练网络
def create_network(content_encoder, decoder):
    with torch.no_grad():
        network = net.Net(content_encoder, decoder)
    network.train()
    network.to(device)

    return network

#加载数据集
def load_dataset(content_dir, style_dir):
    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(content_dir, content_tf)
    style_dataset = FlatFolderDataset(style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    return content_iter, style_iter

#将结果写入tensorboard，便于观察
def tenser_write(writer, network, loss_n, loss_c, loss_s, l_identity1, l_identity2, loss_tv, Iins, loss, Ics, i, loss_ccp):
    if (i+1)%1000 ==0 or (i+1)==160000:
        writer.add_image('content',content_images[0].cpu().detach().numpy(), i + 1)
        writer.add_image('style', style_images[0].cpu().detach().numpy(), i + 1)
        Ics = Ics[0].cpu().detach()
        writer.add_image('Ics',Ics.numpy(),i+1)
    #writer.add_scalar('loss_n', loss_n.data, i + 1)
    writer.add_scalar('loss_content', loss_c.sum().cpu().detach().data, i + 1)
    writer.add_scalar('loss_style', loss_s.sum().cpu().detach().data, i + 1)
    writer.add_scalar('Iins', Iins.sum().cpu().detach().data, i + 1)
    #writer.add_scalar('loss_identity1', l_identity1.data, i + 1)
    #writer.add_scalar('loss_identity2', l_identity2.data, i + 1)
    writer.add_scalar('loss_ccp', loss_ccp.sum().cpu().detach().data, i + 1)
    writer.add_scalar('total_loss', loss.sum().cpu().detach().data, i + 1)

#计算损失函数
def loss_cal(loss_n, loss_c, loss_s, l_identity1, l_identity2, loss_tv, Iins, loss_ccp):
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_tv = 10e-5 * loss_tv
    loss_ccp = loss_ccp * 4
    Iins = 1 * Iins
    loss = 3000 * loss_n + loss_c + loss_s + (l_identity1 * 70 ) + (l_identity2 * 1) + loss_tv + Iins + loss_ccp
    print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
              ,"-l1:",l_identity1.sum().cpu().detach().numpy(),"-l2:",l_identity2.sum().cpu().detach().numpy(),loss_tv.sum().cpu().detach().numpy()
                , "-Iins:", Iins.sum().cpu().detach().numpy(), "-ccp:", loss_ccp.sum().cpu().detach().numpy()
              )

    return loss

def back_step(optimizer, loss):
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

def save_pth(network, i):
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                       i + 1))
        state_dict = network.mcc_module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/mcc_module_iter_{:d}.pth'.format(args.save_dir,
                                                          i + 1))

def train(content_images, style_images, network, optimizer, i):
    #进行训练，返回loss，并根据权重计算出总的loss
    loss_n, loss_c, loss_s, l_identity1, l_identity2, loss_tv, Ics, Iins, loss_ccp = network(content_images, style_images)
    loss = loss_cal(loss_n, loss_c, loss_s, l_identity1, l_identity2, loss_tv, Iins, loss_ccp)
    back_step(optimizer, loss)

    tenser_write(writer, network, loss_n, loss_c, loss_s, l_identity1, l_identity2, loss_tv, Iins, loss, Ics, i, loss_ccp)
    save_pth(network, i)

def create_parser_args():
    # 命令行接口
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
    # training options
    parser.add_argument('--save_dir', default='./models',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--style_weight', type=float, default=18.0)
    parser.add_argument('--content_weight', type=float, default=3.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--use_cuda', type=int, default=1)

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    return args

if __name__ == '__main__':
    cudnn.benchmark = True
    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

    args = create_parser_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.log_dir)


    decoder = net.decoder.to(device)
    style_encoder = net.vgg.to(device)
    style_encoder.load_state_dict(torch.load(args.vgg))
    network = create_network(style_encoder, decoder)

    content_iter, style_iter = load_dataset(args.content_dir, args.style_dir)

    optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                                  {'params': network.mcc_module.parameters()}], lr=args.lr)
    lr = args.lr

    for i in tqdm(range(args.max_iter)):
        if (i+1) % 2000 == 0:
            x = ( i + 1 ) // 2000
            lr = adjust_learning_rate(optimizer, iteration_count=x)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)

        # 训练
        train(content_images, style_images, network, optimizer, i)
        print("learning rate", lr)

    writer.close()

