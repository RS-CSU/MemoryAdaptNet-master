import argparse
import torch
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp

import network
from utils.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.isprs_dataset import ISPRSDataset, ISPRSDataset_val
from metrics import StreamSegMetrics
import utils
import albumentations as albu

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 0

IMAGE_DIRECTORY = "../../dataset/potsdam/train_img_512_1_ori"
LABEL_DIRECTORY = "../../dataset/potsdam/train_lab_512_ori"

IMAGE_DIRECTORY_TARGET = "../../dataset/vaihingen/train_img_512"
LABEL_DIRECTORY_TARGET = "../../dataset/vaihingen/train_lab_512"
VAL_IMAGE_DIRECTORY_TARGET = "../../dataset/vaihingen/val_img_512"
VAL_LABEL_DIRECTORY_TARGET = "../../dataset/vaihingen/val_lab_512"

IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
INPUT_SIZE_TARGET = '512,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 6
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots_p2v_v3_1/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'

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

    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")

    parser.add_argument("--image-dir", type=str, default=IMAGE_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--label-dir", type=str, default=LABEL_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--target-image-dir", type=str, default=IMAGE_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--target-label-dir", type=str, default=LABEL_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--val-target-image-dir", type=str, default=VAL_IMAGE_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--val-target-label-dir", type=str, default=VAL_LABEL_DIRECTORY_TARGET,
                        help="Path to the directory containing the source dataset.")

    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")

    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10
    optimizer.param_groups[2]['lr'] = lr * 10
    optimizer.param_groups[3]['lr'] = lr
    optimizer.param_groups[4]['lr'] = lr * 10
    return lr

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_affine_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),
    ]
    return albu.Compose(train_transform)

def get_color_augmentation():
    train_transform = [
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def validate(args, model, validloader, metrics1, metrics2, lr_backbone):
    """Do validation and return specified samples"""
    metrics1.reset()
    metrics2.reset()
    i_iter  = 0
    with torch.no_grad():
        for i, data in enumerate(validloader, start=0):
            images, labels = data[0], data[1]
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            m,outputs1, outputs2 = model(images, 'TEST', labels, lr_backbone, i_iter)
            preds1 = (outputs1 + outputs2).detach().max(dim=1)[1].cpu().numpy()
            preds2 = outputs2.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics1.update(targets, preds1)
            metrics2.update(targets, preds2)

        score1 = metrics1.get_results()
        score2 = metrics2.get_results()
    return score1, score2

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

    model = model_map[args.model](num_classes=args.num_classes, output_stride=args.output_stride)
    if args.separable_conv and 'plus' in args.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    args_path = os.path.join(args.snapshot_dir, 'args.txt')
    fh = open(args_path, 'a')
    fh.write(str(model))
    fh.write(str(args))
    fh.close()

    model.train()
    model.cuda(args.gpu)
    cudnn.benchmark = True

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    model_D.train()
    model_D.cuda(args.gpu)

    # dataset
    train_dataset = ISPRSDataset(
        args.image_dir,
        args.label_dir,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
        augmentation=get_affine_augmentation()
    )
    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = iter(trainloader)

    target_dataset = ISPRSDataset(
        args.target_image_dir,
        args.target_label_dir,
        max_iters=args.num_steps * args.iter_size * args.batch_size,
        augmentation=get_affine_augmentation()
    )
    targetloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    targetloader_iter = iter(targetloader)

    val_target_dataset = ISPRSDataset_val(
        args.val_target_image_dir,
        args.val_target_label_dir,
    )
    val_targetloader = data.DataLoader(val_target_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, pin_memory=True)

    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': args.learning_rate},
        {'params': model.bottleneck.parameters(), 'lr': 10 * args.learning_rate},
        {'params': model.decoder_stage1.parameters(), 'lr': 10 * args.learning_rate},
        {'params': model.memory_module.parameters(), 'lr': args.learning_rate},
        {'params': model.classifier.parameters(), 'lr':  10 * args.learning_rate},
    ], lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    metrics1 = StreamSegMetrics(args.num_classes)
    metrics2 = StreamSegMetrics(args.num_classes)
    acc_path = os.path.join(args.snapshot_dir, 'acc.txt')
    acc_iter_path = os.path.join(args.snapshot_dir, 'acc_iter.txt')
    best_score = 0.0

    loss_seg_value = 0
    loss_adv_target_value = 0
    loss_D_value = 0

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        lr_backbone = adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(args.iter_size):
            # train G
            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False
            # train with source
            images, labels, path = trainloader_iter.next()
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            #stage 1:only update domain feature alighment brance
            if i_iter < 3000:
                memory, pred1, pred2 = model(images, 'TRAIN', labels, lr_backbone, i_iter)
                loss_seg_stage1 = loss_calc(pred1, labels, args.gpu)
                loss_seg_stage1.backward()
                loss = loss_seg_stage1
            else:
                memory, pred1, pred2 = model(images, 'TRAIN', labels, lr_backbone, i_iter)
                loss_seg_stage1 = loss_calc(pred1, labels, args.gpu)
                loss_seg_stage1.backward(retain_graph=True)
                loss_seg2 = loss_calc(pred2, labels, args.gpu)
                loss_seg2.backward()
                loss = loss_seg2 + loss_seg_stage1

            # proper normalization
            loss = loss / args.iter_size
            loss_seg_value += loss.data.cpu().numpy() / args.iter_size

            # train with target
            images, labels, _ = targetloader_iter.next()
            images = Variable(images).cuda(args.gpu)
            memory, pred_target1, pred_target2 = model(images, 'TARGET', labels, lr_backbone, i_iter)
            D_out = model_D(F.softmax(pred_target1))
            loss_adv_target = bce_loss(D_out,
                                        Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(
                                            args.gpu))
            loss = loss_adv_target
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value += loss_adv_target.data.cpu().numpy() / args.iter_size

            # train D
            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            pred = pred1.detach()
            D_out = model_D(F.softmax(pred))
            loss_D = bce_loss(D_out,
                               Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu))
            loss_D = loss_D / args.iter_size
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()

            # train with target
            pred_target = pred_target1.detach()
            D_out = model_D(F.softmax(pred_target))
            loss_D = bce_loss(D_out,
                               Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda(args.gpu))
            loss_D = loss_D / args.iter_size
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()

        optimizer.step()
        optimizer_D.step()

        if (i_iter) % 10 == 0:
            print('exp = {}'.format(args.snapshot_dir))
            print(
                'iter = {0:8d}/{1:8d},  loss_seg = {2:.3f} , loss_adv = {3:.3f}  loss_D = {4:.3f}'.format(
                    i_iter, args.num_steps, loss_seg_value / 10,
                                            loss_adv_target_value / 10,  loss_D_value / 10))

            loss_seg_value = 0.0
            loss_adv_target_value = 0.0
            loss_D_value = 0.0

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'p2v_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D.state_dict(),
                       osp.join(args.snapshot_dir, 'p2v_' + str(args.num_steps_stop) + '_D.pth'))
            break

        # val
        if (i_iter + 1) % 300 == 0:
            print("validation...")
            val_score1, val_score2 = validate(
                args, model=model, validloader=val_targetloader, metrics1=metrics1, metrics2=metrics2, lr_backbone = lr_backbone)
            print(metrics2.to_str(val_score2))

            fh = open(acc_iter_path, 'a')
            fh.write('iter:' + str(i_iter))
            fh.write(metrics1.to_str(val_score1))
            fh.write(metrics2.to_str(val_score2))
            fh.close()

            if val_score2['Mean IoU'] > best_score:  # save best model
                best_score = val_score2['Mean IoU']
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'p2v_' + str(best_score) + '.pth'))
                torch.save(model_D.state_dict(),
                           osp.join(args.snapshot_dir, 'p2v_' + str(best_score) + '_D.pth'))
                torch.save(memory, osp.join(args.snapshot_dir, 'p2v_' + str(best_score) + '_m.pth'))
                fh = open(acc_path, 'a')
                fh.write('iter:' + str(i_iter))
                fh.write(metrics1.to_str(val_score1))
                fh.write(metrics2.to_str(val_score2))
                fh.close()

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'p2v_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'p2v_' + str(i_iter) + '_D.pth'))
            torch.save(memory, osp.join(args.snapshot_dir, 'p2v_' + str(i_iter) + '_m.pth'))

if __name__ == '__main__':
    main()
