import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


'''

BasicBlock 클래스는 일반적인 residual block의 구조로 코드가 짜여져있다.

init(), forward()

'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

'''

Bottleneck 클래스는 일반적인 bottleneck residual block의 구조로 코드가 짜여져있다.

init(), forward()

'''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        
        # 이 부분이 bottleneck 구조로 차원 수를 줄여 연산하도록 한다.
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

'''

ResFeature 클래스는 Basicblock 또는 Bottleneck 클래스를 기반으로 특징맵을 추출하도록 코드가 짜여져 있다.

init(), _makelayer(), forward()

'''
class ResFeature(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResFeature, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # ResFeature 모델 클래스의 첫번째 Conv 연산의 출력 벡터 차원을 정의한다.
        self.inplanes = 64
        # dilation을 1로 주어 일반적인 Conv 연산을 수행하도록 한다.
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        
        self.groups = groups
        self.base_width = width_per_group

        # 일반적인 nn.Conv2d, ReLU, MaxPool2d와 동일하다.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # _make_layer 함수를 통해서 block 클래스의 레이러를 쌓아 Conv 연산을 수행하는 레이어를 정의한다.
        # 이때 _make_layer에서 반환하는 block 클래스의 레이어를 몇 개 쌓을지를 layers 리스트로 각각 정해준다.
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        '''
        ResGazeEs 모듈의 파라미터를 Kaiming he 방법으로 초기화 한다.
        주어진 인스턴스가 어떤 클래스 또는 데이터 타입인지 확인하는 함수인 isinstance -> (True, False)를 사용하여 초기화를 다르게 한다.
        즉, Conv 연산과 BatchNorm 연산에 사용되는 가중치 파라미터에 대해서 다른 방법으로 초기화 한다.
        '''
        # Pytorch nn.Module을 상속받는 클래스의 self.modules() 함수는 해당 모델 클래스의 모듈로 정의된 레이어들을 반환한다.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 해당 코드는 레이어에 따라 가중치 파라미터 초기화 방식을 각각 다르게 하는데 이를 찾아볼 필요가 있다.
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677 
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        # 정의한 조건에 맞지 않으면 downsample을 아예 사용하지 않도록 초기에 None으로 선언한다. 
        downsample = None
        # stride가 1이 아니거나 (입력 벡터의 차원)이 생성자에게 주어진 (block의 expansion 값 * plane)과 맞지 않으면 True
        # 즉, 특징 벡터를 점차적으로 Upsample을 해주어야 하는데 upscale factor와 차원의 배수가 맞지 않으면, 1x1 Conv + BatchNorm으로 이를 맞춰 주는 작업이다.
        # 그렇기 때문에 필요하면 하고, 필요 없으면 skip이 가능한 것이다. 
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # block을 파라미터에 따라 생성하여 layers 리스트에 추가한다.
        # 레이어의 입력 차원을 재구성하여  block을 파라미터에 따라 생성한 후 layers 리스트에 추가한다.
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        # 이렇게 추가된 layers의 모든 요소들을 시퀀셜 컨테이너로 변환하여 반환한다. 
        return nn.Sequential(*layers)

    def forward(self, x):
        # 선언한 레이어들을 통한 forward 연산, 일반적인 방식과 동일하다.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


'''

ResGazeEs 클래스는 이미지/ 피쳐맵을 입력으로 받아
theta, pi를 출력하도록 하는 구조로 코드가 작성되어있다. 

init(), forward()

'''
class ResGazeEs(nn.Module):

    def __init__(self, ):
        super(ResGazeEs, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)

        '''
        ResGazeEs 모듈의 파라미터를 Kaiming he 방법으로 초기화 한다.
        주어진 인스턴스가 어떤 클래스 또는 데이터 타입인지 확인하는 함수인 isinstance -> (True, False)를 사용하여 초기화를 다르게 한다.
        즉, Conv 연산과 BatchNorm 연산에 사용되는 가중치 파라미터에 대해서 다른 방법으로 초기화 한다.
        '''

        # Pytorch nn.Module을 상속받는 클래스의 self.modules() 함수는 해당 모델 클래스의 모듈로 정의된 레이어들을 반환한다.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # batch가 작을 경우에 발생하는 배치 정규화의 문제를 해결하기 위해서
            # Kaiming he는 GroupNorm을 제안하였다.
            # (Layer norm : feature dimension끼리의 정규화)
            # (Instance Norm : feature dimension끼리 채널 별로 정규화)
            # GroupNorm은 두 정규화 방법의 중간쯤으로 사용자가 입력한 채널의 수 만큼 묶어서 Instance Norm을 수행한다.
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # nn.AdaptiveAvgPool2d을 통해서 입력된 feature map을 벡터로 만들어준다.
        # 이는 global average pooling과 언듯 유사하게 보이지만,
        # nn.AdaptiveAvgPool2d 연산은 channel 단위는 그대로 두고 연산하기 때문에 H x W 의 사이즈만 변화한다.
        # 즉, 해당 연산의 파라미터로는 출력 피쳐맵의 H x W 가 들어간다.
        x = self.avgpool(x)

        # 또한 B x C x 1 x 1배치 단위로 나누어진 feature map을 배치 단위로 벡터화 한다.
        x = x.view(x.size(0), -1)

        # 이러한 벡터를 입력받아 theta와 pi를 출력하도록 한다.
        x = self.fc(x)
        return x

'''

ResDeconv 클래스는 특징 벡터를 이미지로 복원하여 반환하는 구조로 코드가 작성되었다.

init(), _make_layer(), forward()

'''
class ResDeconv(nn.Module):
    def __init__(self, block):
        self.inplanes=2048
        super(ResDeconv, self).__init__()

        # 베이스 모델을 리스트로 선언하고, 필요한 레이어들을 추가해주는 방식으로 코드가 구현되었다.
        # 이를 nn.Sequential 메소드를 이용하여 시퀀셜 컨테이너로 변환하여 하나의 레이어 블록으로 사용한다.
        model = []
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 256, 2)] # 28
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 128, 2)] # 56
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 64, 2)] # 112
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 32, 2)] # 224
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 16, 2)] # 224
        model += [nn.Conv2d(16, 3, stride=1, kernel_size=1)]

        self.deconv = nn.Sequential(*model)

        '''
        ResDeconv 모듈의 파라미터를 Kaiming he 방법으로 초기화 한다.
        주어진 인스턴스가 어떤 클래스 또는 데이터 타입인지 확인하는 함수인 isinstance -> (True, False)를 사용하여 초기화를 다르게 한다.
        즉, Conv 연산과 BatchNorm 연산에 사용되는 가중치 파라미터에 대해서 다른 방법으로 초기화 한다.
        두 연산에 사용되는 가중치 파라미터에 대해서 다른 초기화 방법을 사용하는거지? 
        '''
        # Pytorch nn.Module을 상속받는 클래스의 self.modules() 함수는 해당 모델 클래스의 모듈로 정의된 레이어들을 반환한다.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # 정의한 조건에 맞지 않으면 downsample을 아예 사용하지 않도록 초기에 None으로 선언한다. 
        downsample = None
        # stride가 1이 아니거나 (입력 벡터의 차원)이 생성자에게 주어진 (block의 expansion 값 * plane)과 맞지 않으면 True
        # 즉, 특징 벡터를 점차적으로 Upsample을 해주어야 하는데 upscale factor와 차원의 배수가 맞지 않으면, Conv + BatchNorm으로 이를 맞춰 주는 작업이다.
        # 그렇기 때문에 필요하면 하고, 필요 없으면 skip이 가능한 것이다. 
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        '''
        Upsample을 위해 필요한 조건을 맞추기 위해 downsample 레이어를 생성하고,
        정의할 본체 레이어를 _make_layer 함수에 주어진 파라미터에 따라 생성하여 추가한다.
        '''
        layers = []
        
        # block을 파라미터에 따라 생성하여 layers 리스트에 추가한다.
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 레이어의 입력 차원을 재구성하여  block을 파라미터에 따라 생성한 후 layers 리스트에 추가한다.
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # 이렇게 추가된 layers의 모든 요소들을 시퀀셜 컨테이너로 변환하여 반환한다. 
        return nn.Sequential(*layers)

    def forward(self, features):
        # 생성된 deconv 레이어를 통해 입력 특징 벡터를 이미지로 반환한다. 
        img = self.deconv(features)
        return img


'''
_resnet 클래스는 주어진 block 클래스를 기반으로 모델을 생성한다.
'''
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResFeature(block, layers, **kwargs)
    # 가중치 load 유무를 체크한다.
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

