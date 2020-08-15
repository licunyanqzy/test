import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import logging

from data.loader import data_loader
from models import TrajectoryPrediction
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
parser.add_argument("--dataset_name", default="zara2", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)    # The number of background threads to use for data loading
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--batch_size", default=64, type=int)

parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)

parser.add_argument("--action_encoder_input_dim", default=8, type=int)
parser.add_argument("--action_encoder_hidden_dim", default=32, type=int)
parser.add_argument("--goal_encoder_input_dim", default=16, type=int)
parser.add_argument("--goal_encoder_hidden_dim", default=32, type=int)
parser.add_argument("--goal_decoder_input_dim", default=16, type=int)
parser.add_argument("--goal_decoder_hidden_dim", default=32, type=int)
parser.add_argument("--action_decoder_input_dim", default=16, type=int)
parser.add_argument("--action_decoder_hidden_dim", default=32, type=int)

parser.add_argument()   # n_units...
parser.add_argument()   # n_heads...
parser.add_argument("--dropout", default=0, type=float) # dropout rate
parser.add_argument("alpha", default=0.2, type=float)   # alpha for the leaky relu

parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--num_epoch", default=400, type=int)

parser.add_argument("--gpu_num", default="1", type=str)
parser.add_argument("--use_gpu", default=True, type=bool)

parser.add_argument("--resume", default="", type=str)


bestADE = 100


def train(args, model, train_loader, optimizer, epoch, writer, training_step):
    losses = utils.AverageMeter("Loss", ":.6f")     # 作用 ?
    progress = utils.ProgressMeter(                 # 作用 ?
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,           # obs_traj, [8,1413,2]
            pred_traj_gt,       # pred_traj, [12,1413,2]
            obs_traj_rel,      # obs_traj_rel, [8,1413,2]
            pred_traj_gt_rel,   # pred_traj_rel, [12, 1413,2]
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ) = batch

        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len :]

        # 缺少标注数据的模块,...

        if training_step == 1:

        elif training_step == 2:



def validate(args, model, val_loader, epoch, writer):

    return


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = utils.get_dset_path(args.dataset_name, "train")
    val_path = utils.get_dset_path(args.dataset_name, "test")

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    writer = SummaryWriter()

    model = TrajectoryPrediction(   # 实例化模型,...
        obs_len=args.obs_len,
        pred_len=args.pred_len,

    )
    model.cuda()
    optimizer = optim.Adam(     # 优化参数,...
        # model.parameters()
        # 是否需要为每个参数单独设置 ?
        lr=args.lr
    )

    if args.resume:     # start from checkpoint
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch=checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found as '{}'".format(args.resume))

    global bestADE

    for epoch in range(args.start_epoch, args.num_epoch):   # training...
        if epoch < 150:
            train(args, model, train_loader, optimizer, epoch, writer, 1)
        else:
            train(args, model, train_loader, optimizer, epoch, writer, 2)

            ADE = validate(args, model, val_loader, epoch, writer)
            isBest = ADE > bestADE
            bestADE = min(ADE, bestADE)

            utils.save_checkpoint(  # if ADE > bestADE, save checkpoint
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ADE": bestADE,
                    "optimizer": optimizer.state_dict(),
                },
                isBest,
                f"./checkpoint/checkpoint{epoch}.pth.tar",
            )

    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    checkpoint_dir = "./checkpoint"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)


