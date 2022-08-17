import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy

from utils.models.memorynet.memory import FeaturesMemory
from utils.backbones import BuildActivation, BuildNormalization

update_cfg: {
            'strategy': 'cosine_similarity',
            'ignore_index': 255,
            'momentum_cfg': {
                'base_momentum': 0.9,
                'base_lr': 0.01,
                'adjust_by_learning_rate': True,
            }
        }
act_cfg: {'inplace': True}
class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, num_classes):
        super(_SimpleSegmentationModel, self).__init__()
        #memory bank
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder_stage1 = nn.Sequential(
            # nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0,
            #           bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1),
        )
        self.memory_module = FeaturesMemory(
            num_classes=num_classes,
            feats_channels=512,
            transform_channels=256,
            num_feats_per_cls=1,
            out_channels=2048,
            use_context_within_image=True,
            use_hard_aggregate=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.backbone = backbone
        self.classifier = classifier
    def forward(self, x, mode, target, lr):
        img_size = x.size(2), x.size(3)
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        #feed to memory
        #torch.Size([B, 512, 32, 32])
        memory_input = self.bottleneck(features['out'])
        #torch.Size([2, 6, 32, 32])
        preds_stage1 = self.decoder_stage1(memory_input)
        tored_memory, memory_output = self.memory_module(memory_input, preds_stage1)

        x = self.classifier(memory_output)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        if mode == 'TRAIN':
            #updata memory
            with torch.no_grad():
                self.memory_module.update(
                    features=F.interpolate(memory_input, size=img_size, mode='bilinear', align_corners=False),
                    segmentation=target,
                    learning_rate=lr,
                    strategy = 'cosine_similarity',
                    ignore_index = 255,
                    base_momentum = 0.9,
                    base_lr = 0.01,
                    adjust_by_learning_rate = True,
                )
        preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=False)
        return preds_stage1, x

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
