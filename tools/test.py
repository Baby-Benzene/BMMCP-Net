import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.model_pvt_v2_b1 import BaseNet

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler
import torch.nn as nn

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
from PIL import Image

import os, time
import numpy as np
from argparse import ArgumentParser


criterion_bce = nn.BCEWithLogitsLoss()


def BCEDiceLoss(inputs, targets):
    bce = criterion_bce(inputs, targets)
    probs = torch.sigmoid(inputs)
    inter = (probs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (probs.sum() + targets.sum() + eps)
    return bce + 1 - dice


def BCE(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    return bce


@torch.no_grad()
def val(args, val_loader, model, epoch):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    epoch_loss = []

    total_batches = len(val_loader)
    print(f"Total test batches: {total_batches}")

    for iter, batched_inputs in enumerate(val_loader):

        img, target = batched_inputs
        img_name = val_loader.sampler.data_source.file_list[iter]
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the model
        output, output2, output3, output4 = model(pre_img_var, post_img_var)


        loss_main = BCEDiceLoss(output, target_var)
        loss_m2 = BCEDiceLoss(output2, target_var)
        loss_m3 = BCEDiceLoss(output3, target_var)
        loss_m4 = BCEDiceLoss(output4, target_var)
        loss = 1.0 * loss_main + 0.4 * loss_m2 + 0.2 * loss_m3 + 0.1 * loss_m4

        pred = torch.where(output > 0, torch.ones_like(output), torch.zeros_like(output)).long()

        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        pr = pred[0, 0].cpu().numpy()
        gt = target_var[0, 0].cpu().numpy()
        index_tp = np.where(np.logical_and(pr == 1, gt == 1))
        index_fp = np.where(np.logical_and(pr == 1, gt == 0))
        index_tn = np.where(np.logical_and(pr == 0, gt == 0))
        index_fn = np.where(np.logical_and(pr == 0, gt == 1))

        map = np.zeros([gt.shape[0], gt.shape[1], 3])
        map[index_tp] = [255, 255, 255]  # white
        map[index_fp] = [255, 0, 0]  # red
        map[index_tn] = [0, 0, 0]  # black
        map[index_fn] = [0, 255, 255]  # Cyan

        change_map = Image.fromarray(np.array(map, dtype=np.uint8))
        change_map.save(os.path.join(args.vis_dir, img_name))

        f1 = salEvalVal.update_cm(pr, gt)

        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f loss: %.3f time: %.3f' % (iter, total_batches, f1, loss.data.item(), time_taken),
                  end='')

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def ValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = BaseNet()

    args.savedir = args.savedir + '_' + args.file_root + '_batch_' + str(args.train_batch_size) + '_lr_' + str(
        args.lr) + '/'
    args.vis_dir = './Predict/' + args.file_root + '/'

    if args.file_root == 'LEVIR':
        args.file_root = 'C:/CD-D/LEVIR'
    elif args.file_root == 'SYSU':
        args.file_root = 'C:/CD-D/SYSU-O'
    elif args.file_root == 'BCDD':
        args.file_root = 'C:/CD-D/BCDD'
    elif args.file_root == 'WHU':
        args.file_root = 'C:/CD-D/WHU-CD'
    elif args.file_root == 'quick_start':
        args.file_root = './samples'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.savedir):
        print(f"Warning: Model directory {args.savedir} not found! Please check arguments.")
        os.makedirs(args.savedir, exist_ok=True)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa', 'IoU', 'F1', 'R', 'P'))
    logger.flush()

    # load the model
    model_file_name = args.savedir + 'best_model.pth'
    if not os.path.exists(model_file_name):
        raise FileNotFoundError(
            f"Model file not found at {model_file_name}. Ensure it was trained and saved correctly.")

    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)
    print(f"=> Loaded model from {model_file_name}")

    loss_test, score_test = val(args, testLoader, model, 0)
    print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('Test', score_test['Kappa'], score_test['IoU'],
                                                                   score_test['F1'], score_test['recall'],
                                                                   score_test['precision']))
    logger.flush()
    logger.close()

    import scipy.io as scio

    scio.savemat(args.vis_dir + 'results.mat', score_test)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR", help='Data directory | LEVIR | BCDD | SYSU ')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size used during training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate (must match training)')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--logFile', default='testLog.txt', help='File that stores the test logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    ValidateSegmentation(args)