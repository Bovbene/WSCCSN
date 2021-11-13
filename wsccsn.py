from model import common

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


def make_model(args, parent=False):
    return WSCCSN(args)

def load_wcoeff(path):
    from scipy import io as scio
    data = scio.loadmat(path)['wcoeff']
    wcoeff_dict = {}
    wcoeffs = data[0][0].tolist()
    names = [name[0] for name in data.dtype.descr]
    for name, wcoeff in zip(names, wcoeffs):
        wcoeff_dict[name] = wcoeff
    return wcoeff_dict

def dwtmtx(N, wtype='haar', wlev=1):
    wcoeff_dict = load_wcoeff('./wcoeff.mat')
    h, g = wcoeff_dict[wtype].tolist()
    L = len(h)
    h_l = h[::-1]
    g_l = g[::-1]
    from math import log2
    loop_max = log2(N)
    loop_min = log2(L)
    if wlev > loop_max - loop_min:
        from warnings import warn
        warn("\nWaring: wlev is too big \nThe biggest wlev is {}"
             .format(loop_max - loop_min + 1), DeprecationWarning)
    ww = 1
    for loop in range(int(loop_max - wlev + 1), int(loop_max + 1)):
        Nii = 2 ** loop
        p1_0 = np.r_[h_l, np.zeros(Nii - L)]
        p2_0 = np.r_[g_l, np.zeros(Nii - L)]
        p1 = np.zeros((Nii // 2, Nii))
        p2 = np.zeros((Nii // 2, Nii))
        for ii in range(1, Nii // 2 + 1):
            p1[ii - 1:] = np.roll(p1_0, int(2 * (ii - 1) + 1 - (L - 1) + L / 2 - 1))
            p2[ii - 1:] = np.roll(p2_0, int(2 * (ii - 1) + 1 - (L - 1) + L / 2 - 1))
        w1 = np.r_[p1, p2]
        mm = int(2 ** loop_max - len(w1))
        w = np.vstack([np.hstack((w1, np.zeros((len(w1), mm)))), np.hstack((np.zeros((mm, len(w1))), np.eye(mm)))])
        if type(ww) is int:
            ww = ww * w
        elif type(ww) is np.ndarray:
            ww = ww @ w
    return ww


class ClippedReLU(nn.Module):
    def __init__(self):
        super(ClippedReLU, self).__init__()

    def forward(self, x):
        return x.clamp(min=0., max=255.)

class Gblock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(Gblock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class BranchLL(nn.Module):
    def __init__(self, SR_rate):
        from math import sqrt
        super(BranchLL, self).__init__()

        self.conv0_0 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_1 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_2 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_3 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = ConvRelu(in_channels=48, out_channels=32, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3 * SR_rate ** 2, kernel_size=3, padding=1)

        self.Gblocks = nn.Sequential(Gblock(32, 32, 4), Gblock(32, 32, 4), Gblock(32, 32, 4))
        # self.depth2spcae = nn.PixelShuffle(SR_rate)
        self.clippedReLU = nn.PReLU()

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                std = sqrt(2 / fan_out * 0.1)
                torch.nn.init.normal_(m.weight.data, mean=0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)

    def forward(self, x):

        res_conv0_0 = self.conv0_0(x)
        res_conv0_1 = self.conv0_1(x)
        res_conv0_2 = self.conv0_2(x)
        res_conv0_3 = self.conv0_3(x)
        res = torch.cat((res_conv0_0, res_conv0_1, res_conv0_2, res_conv0_3), dim=1)

        res = self.conv2(res)
        res = self.Gblocks(res)

        res_conv1 = self.conv1(x)
        res = torch.cat((res, res_conv1), dim=1)

        res = self.conv3(res)
        res = self.conv4(res)
        res = self.clippedReLU(res)

        # res = self.depth2spcae(res)

        return res


class BranchLL_quantization(nn.Module):
    def __init__(self, SR_rate):
        super(BranchLL_quantization, self).__init__()
        from torch.quantization import QuantStub, DeQuantStub
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.conv0_0 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_1 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_2 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)
        self.conv0_3 = ConvRelu(in_channels=3, out_channels=8, kernel_size=3)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = ConvRelu(in_channels=48, out_channels=32, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3 * SR_rate ** 2, kernel_size=3, padding=1)

        # self.Gblocks = nn.Sequential(Gblock(32, 32, 4), Gblock(32, 32, 4), Gblock(32, 32, 4))
        self.Gblocks = nn.Sequential(Gblock(32, 32, 4))
        # self.depth2spcae = nn.PixelShuffle(SR_rate)
        self.clippedReLU = ClippedReLU()
        self.cat1 = nn.quantized.FloatFunctional()
        self.cat2 = nn.quantized.FloatFunctional()

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)

    def forward(self, x):

        x = self.quant(x)
        res_conv0_0 = self.conv0_0(x)
        res_conv0_1 = self.conv0_1(x)
        res_conv0_2 = self.conv0_2(x)
        res_conv0_3 = self.conv0_3(x)

        res = self.cat1.cat((res_conv0_0, res_conv0_1, res_conv0_2, res_conv0_3), dim=1)

        res = self.conv2(res)
        res = self.Gblocks(res)

        res_conv1 = self.conv1(x)
        res = self.cat2.cat((res, res_conv1), dim=1)

        res = self.conv3(res)
        res = self.conv4(res)

        # res = self.depth2spcae(res)
        res = self.clippedReLU(res)

        res = self.dequant(res)
        return res

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvRelu:
                # print("fuse conv and relu in ConvRelu")
                torch.quantization.fuse_modules(m, ['conv', 'relu'], inplace=True)
            if type(m) == Gblock:
                # print("fuse conv and relu in Gblock")
                torch.quantization.fuse_modules(m, ['conv0', 'relu'], inplace=True)

class BranchH(nn.Module):

    def __init__(self,args,conv=common.default_conv):
        super(BranchH, self).__init__()
        n_resblocks = 4
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        # act = nn.ReLU(True)
        act = nn.PReLU()

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.softmax = nn.Softmax(-1)
        # define body module

        m_body = [
            common.DCNResBlock(
                common.dcnv2,conv, n_feats, kernel_size, act=nn.Softmax(-1), res_scale=1
            )
        ]+[
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [conv(n_feats, args.n_colors, kernel_size)]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x



class WSCCSN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(WSCCSN, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        #act = nn.ReLU(True)
        act = nn.PReLU()
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        self.url = None
        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors*4, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors*4, kernel_size)
        ]
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.branchll = BranchLL(1)
        self.branchh = BranchH(args)
        self.wfilter = torch.tensor(dwtmtx(self.patch_size // 4), dtype=torch.float32).expand(self.batch_size,
                                                                                              self.patch_size // 4,
                                                                                              self.patch_size // 4).cuda()
        self.wfilter2 = torch.tensor(dwtmtx(self.patch_size), dtype=torch.float32).expand(self.batch_size,
                                                                                          self.patch_size,
                                                                                          self.patch_size).cuda()

    def forward(self, x):
        batch_size, _, height, width = x.size()
        if height != self.patch_size//4 or width != self.patch_size//4:# or self.is_test:
            length = max([height,width])
            from math import log2
            N = log2(length)
            N = 2 ** (int(N) + 1)
            self.wfilter = torch.tensor(dwtmtx(N),dtype=torch.float32).expand(batch_size,N,N).cuda()
            self.wfilter2 = torch.tensor(dwtmtx(N*4), dtype=torch.float32).expand(batch_size, N*4, N*4).cuda()
            # print(x.size(),0,N-width,0,N-height)
            x = F.pad(x,(0,N-width,0,N-height),"constant",value=0)
        else:
            self.wfilter = torch.tensor(dwtmtx(self.patch_size // 4), dtype=torch.float32).expand(batch_size,self.patch_size // 4,self.patch_size // 4).cuda()
            self.wfilter2 = torch.tensor(dwtmtx(self.patch_size), dtype=torch.float32).expand(batch_size,self.patch_size,self.patch_size).cuda()

        # x = self.sub_mean(x)
        r, g, b = x[:,0,:,:], x[:,1,:,:], x[:,2,:,:]
        # print(x.size(),self.wfilter.size())
        dwt = lambda x: torch.bmm(torch.bmm(self.wfilter,x),self.wfilter.transpose(2,1))
        rx, gx, bx = dwt(r), dwt(g), dwt(b)
        def get_sub(code):
            height, width = code.shape[1], code.shape[2]
            if height != width:
                print(height, width)
                raise ValueError("Input doesn't match")
            ll, lh = code[:,:height // 2, :width // 2], code[:,:height // 2, width // 2:]
            hl, hh = code[:,height // 2:, :width // 2], code[:,height // 2:, width // 2:]
            return ll, lh, hl, hh

        rx = torch.stack(get_sub(rx), dim=1)
        gx = torch.stack(get_sub(gx), dim=1)
        bx = torch.stack(get_sub(bx), dim=1)
        x = torch.cat([rx, gx, bx], dim=1)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(x)
        # x = self.add_mean(x)

        rx, gx, bx = x[:,0:4,:, :], x[:,4:8,:, :], x[:,8:12,:, :]

        restructure = lambda i: torch.stack([rx[:,i,:,:],gx[:,i,:,:],bx[:,i,:,:]], dim=1)
        ll, lh, hl, hh = [restructure(i) for i in range(4)]

        ll = self.branchll(ll)
        lh = self.branchh(lh)
        hl = self.branchh(hl)
        hh = self.branchh(hh)

        inv_restructure = lambda i: torch.stack([ll[:,i,:,:],lh[:,i,:,:],hl[:,i,:,:],hh[:,i,:,:]], dim=1)
        rx,gx,bx = [inv_restructure(i) for i in range(3)]

        def get_merge(code):
            ll, lh, hl, hh = code[:,0,:, :],code[:,1,:, :],code[:,2,:, :],code[:,3,:, :]
            row1 = torch.cat([ll,lh],dim = 2)
            row2 = torch.cat([hl, hh], dim = 2)
            code = torch.cat([row1,row2],dim = 1)
            return code#.view(4,1,256,256)
        rx,bx,gx = get_merge(rx),get_merge(bx),get_merge(gx)

        idwt = lambda x: torch.bmm(torch.bmm(self.wfilter2.transpose(2, 1), x), self.wfilter2)

        r, g, b = idwt(rx), idwt(gx), idwt(bx)
        y = torch.stack([r, g, b], dim=1)

        if height != self.patch_size // 4 or width != self.patch_size // 4:# or self.is_test:
            y = y[:,:,:height*4,:width*4]

        return y

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

