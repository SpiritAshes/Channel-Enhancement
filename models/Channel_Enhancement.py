
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch

from models.ECA import ECAAttention
from models.rep import RepVGGBlock, repvgg_model_convert

class Channel_Enhancement(torch.nn.Module):

    def __init__(self, deploy=False):
        super(Channel_Enhancement, self).__init__()
        c1, c2, c3, c4, c5, d1 = 32, 64, 128, 256, 64, 256
        det_h = 65

        self.deploy = deploy
        self.relu = torch.nn.ReLU(inplace=True)

        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.conv1 = RepVGGBlock(1, c1, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=False)

        self.conv2 = RepVGGBlock(c1, c1, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)

        self.conv3 = RepVGGBlock(c1, c2, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=False)

        self.conv4 = RepVGGBlock(c2, c2, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)

        self.conv5 = RepVGGBlock(c2, c3, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=False)

        self.conv6 = RepVGGBlock(c3, c3, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)

        self.conv7 = RepVGGBlock(c3, c4, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)

        self.pixel_shuffle = torch.nn.PixelShuffle(2)

        self.conv_attention = ECAAttention(3)

        self.convS_1 = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnS_1 = nn.BatchNorm2d(det_h)
        self.convS_2 = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnS_2 = nn.BatchNorm2d(d1)

        self.convDF_1 = torch.nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1)
        self.bnDF_1 = nn.BatchNorm2d(c5)
        self.convDF_2 = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnDF_2 = nn.BatchNorm2d(det_h)

        self.convDD_1 = torch.nn.Conv2d(c5, d1, kernel_size=3, stride=1, padding=1)
        self.bnDD_1 = nn.BatchNorm2d(d1)
        self.convDD_2 = torch.nn.Conv2d(d1, d1, kernel_size=1, stride=1, padding=0)
        self.bnDD_2 = nn.BatchNorm2d(d1)


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.pixel_shuffle(x)
        x = self.pool(x)

        x_1 = self.bnS_1(self.convS_1(x))
        x_2 = self.bnS_2(self.convS_2(x))

        x = self.relu(self.conv_attention(x))

        Head_DF = self.relu(self.bnDF_1(self.convDF_1(x)))
        semi = self.bnDF_2(self.convDF_2(Head_DF))
        semi = semi + x_1

        Head_DD = self.relu(self.bnDD_1(self.convDD_1(x)))
        desc = self.bnDD_2(self.convDD_2(Head_DD))
        desc = desc + x_2
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        return semi, desc



def main(weights_path, save_path):
  
  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  net = Channel_Enhancement()
  checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
  net.load_state_dict(checkpoint['model_state_dict'])
  deploy_net = repvgg_model_convert(net, save_path=save_path)
  print('==> Successfully deploy network.')
  return deploy_net

if __name__ == '__main__':
    main('CE.pth.tar','./weight/CE_weight')
