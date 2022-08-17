import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from collections import OrderedDict
from utils.models.memorynet.memory import FeaturesMemory
from utils.network._deeplab import ASPP

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    b, c, h, w = prob.size()
    return -torch.mul(prob, torch.log(prob + 1e-30)) / np.log(c)

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, num_classes, aspp_dilate=[12, 24, 36], memory=None):
        super(_SimpleSegmentationModel, self).__init__()
        # memory bank
        self.decoder_stage1 = nn.Sequential(
            ASPP(2048, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self.bottleneck = nn.Sequential(
            # ASPP(2048, aspp_dilate,out_channels = 512),
            nn.Conv2d(2048, 1024, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.memory_module = FeaturesMemory(
            num_classes=num_classes,
            feats_channels=1024,
            transform_channels=512,
            num_feats_per_cls=1,
            out_channels=2048,
            use_context_within_image=True,
            use_hard_aggregate=False,
            memory_data=memory
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x, mode, target, lr, i_iter):
        img_size = x.size(2), x.size(3)
        input_shape = x.shape[-2:]
        #feature:torch.Size([4, 2048, 32, 32])
        features = self.backbone(x)
        # feed to memory
        preds_stage1 = self.decoder_stage1(features['out'])
        preds_stage2 = None
        tored_memory = None

        #迭代次数大于等于3000次的时候，开始更新memory，保证此时的特征是不变特征。
        if i_iter >= 3000 or mode == 'TEST':
            # memory_output:torch.Size([4, 256, 32, 32])
            memory_input = self.bottleneck(features['out'])
            tored_memory, memory_output = self.memory_module(memory_input, preds_stage1)
            # x:torch.Size([4, 6, 32, 32])
            preds_stage2 = self.classifier(memory_output)
            preds_stage2 = F.interpolate(preds_stage2, size=input_shape, mode='bilinear', align_corners=False)

            if mode == 'TRAIN':
                # updata memory
                with torch.no_grad():
                    self.memory_module.update(
                        features=F.interpolate(memory_input, size=img_size, mode='bilinear', align_corners=False),
                        segmentation=target,
                        learning_rate=lr,
                        strategy='cosine_similarity',
                        ignore_index=255,
                        base_momentum=0.9,
                        base_lr=0.01,
                        adjust_by_learning_rate=True,
                    )
            if mode == 'TARGET':
                    # updata memory
                    target_tar = preds_stage2.detach().max(dim=1)[1]
                    entropy = prob_2_entropy(F.softmax(preds_stage2.detach()))
                    entropy = torch.sum(entropy, axis=1)  # 2,512,512
                    # # # #
                    # # #高斯爬升曲线参数
                    # t = i_iter * 10e-5
                    # arfa = (1 - math.exp(-0.05 * t)) / (1 + math.exp(-0.05 * t))
                    arfa = 0.3
                    target_tar[entropy > arfa] = 255
                    with torch.no_grad():
                        self.memory_module.update(
                            features=F.interpolate(memory_input, size=img_size, mode='bilinear', align_corners=False),
                            segmentation=target_tar,
                            learning_rate=lr,
                            strategy='cosine_similarity',
                            ignore_index=255,
                            base_momentum=0.9,
                            base_lr=0.01,
                            adjust_by_learning_rate=True,
                        )
        preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=False)
        return tored_memory, preds_stage1, preds_stage2

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
