import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import network
from dataset.isprs_dataset import ISPRSDataset, ISPRSDataset_val
from metrics import StreamSegMetrics
import utils
import util.util as util
from util.visualizer import Visualizer
from util import html
from collections import OrderedDict
import os

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 0
VAL_IMAGE_DIRECTORY_TARGET = "./dataset/vaihingen/val_img_512"
VAL_LABEL_DIRECTORY_TARGET = "./dataset/vaihingen/val_lab_512"
IGNORE_LABEL = 255
NUM_CLASSES = 6

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default='deeplabv3_resnet101',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--val-target-image-dir", type=str, default=VAL_IMAGE_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--val-target-label-dir", type=str, default=VAL_LABEL_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()

def main():
    """Create the model and start the training."""

    cudnn.enabled = True

    # Create network
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    save_path_m = './snapshots/p2v_best_m.pth'
    memory = torch.load(save_path_m)
    model = model_map[args.model](num_classes=args.num_classes, output_stride=args.output_stride, memory=memory)
    if args.separable_conv and 'plus' in args.model:
        network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)
    save_path = './snapshots/p2v_best.pth'
    model.load_state_dict(torch.load(save_path))
    print('load success')

    model.eval()
    model.cuda(args.gpu)
    cudnn.benchmark = True

    val_target_dataset = ISPRSDataset_val(
        args.val_target_image_dir,
        args.val_target_label_dir,
    )
    val_targetloader = torch.utils.data.DataLoader(val_target_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, pin_memory=True)

    # implement model.optim_parameters(args) to handle different models' lr setting
    metrics2 = StreamSegMetrics(args.num_classes)
    visualizer = Visualizer(args)
    web_dir = os.path.join('results/results_p2v_1')
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % ('resnet_gan', 'test', 'latest'))

    with torch.no_grad():
        for i, data in enumerate(val_targetloader, start=0):
            images, labels, path = data[0], data[1], data[2]
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            memory, outputs1, outputs2 = model(images, 'TEST', labels, 0, 0)

            for b in range(args.batch_size):
                label_t = torch.unsqueeze(labels[b, :, :], 0)
                output_t = outputs2[b, :, :, :]
                image_t = images[b, :, :, :]
                visuals = OrderedDict([('input_label', util.tensor2label(label_t, args.num_classes)),
                                       ('synthesized_label', util.tensor2label(output_t, args.num_classes)),
                                       ('real_image', util.tensor2im(image_t))])
                img_path = path[b]
                print('process image... %s' % img_path)
                visualizer.save_images(webpage, visuals, img_path)

            preds2 = outputs2.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics2.update(targets, preds2)

        score2 = metrics2.get_results()
        print(score2)
if __name__ == '__main__':
    main()
