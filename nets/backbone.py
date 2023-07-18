import torch
import torch.nn as nn

from .attention import se_block, cbam_block, eca_block, CoordAtt, SOCA

attention_blocks = [se_block, cbam_block, eca_block, CoordAtt, SOCA]

from .model1 import swin_transformer_v2_b

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

#houjia
class MyConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(MyConv, self).__init__()
#深度可分离卷积
        self.conv1 = nn.Conv2d(c1, c1, 3, s, autopad(3, p), groups=c1, bias=False)
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)

        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(self.conv1(x))))

    def fuseforward(self, x):
        return self.act(self.conv(x))

#houjia
import torch.nn.functional as F
class MyMulti_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[1], act=SiLU()):
        super(MyMulti_Concat_Block, self).__init__()
        c_ = int(c2 * e)

        self.ids = ids

        self.dw2 = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.con1_1 = nn.Conv2d(c1, c_, 1, 1, 0, groups=1, bias=False)

        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        #self.cv1 = MyConv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2)+ c1, c3, 1, 1)

    def forward(self, x):
        m1 = self.act(self.bn(self.con1_1(self.dw2(x))))

     #   x_1 = self.cv1(x)
        x_2 = self.cv2(x)

        x_all = [m1, x_2]
     #   x_all = [x_1, x_2]
        # [-1, -3, -5, -6] => [5, 3, 1, 0]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)

        out = self.cv4(torch.cat(tuple([x_all[id] for id in self.ids]) + (x,), dim=1))

       # out = self.cv4(torch.cat([x_all[id] for id in self.ids], [x]))

        return out
#HOUJIA
class My1Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(My1Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)

        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)
        self.cv5 = MyConv(c1, c1, 1, 1)

    def forward(self, x):
        x = self.cv5(x)
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        x = self.cv5(x)

        x_all = [x_1, x_2]
        # [-1, -3, -5, -6] => [5, 3, 1, 0]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)

        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))


       # out2 = torch.cat(out,x)
        return out + x


class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)
        
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        
        x_all = [x_1, x_2]
        # [-1, -3, -5, -6] => [5, 3, 1, 0]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
            
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)
    
class Transition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(Transition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)
        
        self.mp  = MP()

    def forward(self, x):
        # 160, 160, 256 => 80, 80, 256 => 80, 80, 128
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)
        
        # 160, 160, 256 => 160, 160, 128 => 80, 80, 128
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)
        
        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        return torch.cat([x_2, x_1], 1)

#houjia
class MyTransition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(MyTransition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = MyConv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)

        self.mp = MP()

    def forward(self, x):
        # 160, 160, 256 => 80, 80, 256 => 80, 80, 128
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)

        # 160, 160, 256 => 160, 160, 128 => 80, 80, 128
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)

        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        return torch.cat([x_2, x_1], 1)

#houjia
class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
#houjia
class MyC2f1(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#刚加
        conv1 = nn.Conv2d(self.c, self.c, 3, s, autopad(3, p), groups=self.c, bias=False)
        conv   = nn.Conv2d(self.c, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv2 = nn.Sequential(conv1, conv)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c, self.c), 1))
        m1 = self.conv2(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat([y, m1], 1))


class Backbone(nn.Module):
    def __init__(self, transition_channels, block_channels, base_depth, n, phi, pretrained=False, phiii=4):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#
        ids = {
            'l' : [-1, -3, -5, -6],
            'x' : [-1, -3, -5, -7, -8], 
        }[phi]
        # 640, 640, 3 => 640, 640, 32 => 320, 320, 64
        self.stem = nn.Sequential(
            Conv(3, transition_channels, 3, 1),
            Conv(transition_channels, transition_channels * 2, 3, 2),
            Conv(transition_channels * 2, transition_channels * 2, 3, 1),
        )
        # 320, 320, 64 => 160, 160, 128 => 160, 160, 256
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids),
            #MyConv(transition_channels * 4,transition_channels * 8,1,1),
       # )
            #MyC2f(transition_channels * 4, transition_channels * 8, base_depth * 2, True),
        )
        # 160, 160, 256 => 80, 80, 256 => 80, 80, 512
        self.dark3 = nn.Sequential(
            Transition_Block(transition_channels * 8, transition_channels * 4),
            Multi_Concat_Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids), #输出通道数减半
            #MyConv(transition_channels * 8, transition_channels * 16, 1, 1),
       # )
            #C2f(transition_channels * 8, transition_channels * 16, base_depth * 2, True),
        )
        # 80, 80, 512 => 40, 40, 512 => 40, 40, 1024
        self.dark4 = nn.Sequential(
            Transition_Block(transition_channels * 16, transition_channels * 8),
            MyMulti_Concat_Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids),
           # MyConv(transition_channels * 16, transition_channels * 32, 1, 1),

            #MyC2f(transition_channels * 16, transition_channels * 32, base_depth * 2, True),
        )
        # 40, 40, 1024 => 20, 20, 1024 => 20, 20, 1024
        self.dark5 = nn.Sequential(
            Transition_Block(transition_channels * 32, transition_channels * 16),

            MyMulti_Concat_Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids),

            #swin_transformer_v2_b()
            #Multi_Concat_Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids),
           # MyConv(transition_channels * 32, transition_channels * 32, 1, 1),

           # C2f(transition_channels * 32, transition_channels * 32, base_depth, True),
        )

        self.phiii = phiii
        if phiii >= 1 and phiii <= 5:
            self.dark2_attention = attention_blocks[phiii - 1](256)

        if pretrained:
            url = {
                "l" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)

     #   if self.phiii > 1 and self.phiii <= 5:
     #       x = self.dark2_attention(x)

        #-----------------------------------------------#
        #   dark3的输出为80, 80, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3
