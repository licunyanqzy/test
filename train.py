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
from utils import int_tuple


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
parser.add_argument("--alpha", default=0.2, type=float)   # alpha for the leaky relu
parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian", type=str)

parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--num_epoch", default=400, type=int)

parser.add_argument("--gpu_num", default="1", type=str)
parser.add_argument("--use_gpu", default=True, type=bool)
parser.add_argument("--print_every", default=10, type=int)

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

        # 标注 action, goal
        obs_action = utils.cal_action(obs_traj_rel)
        obs_goal = utils.cal_goal(obs_traj_rel, obs_action)
        pred_action_gt = utils.cal_action(pred_traj_gt_rel)
        pred_goal_gt = utils.cal_goal(pred_traj_gt_rel, pred_action_gt)

        input_traj = torch.cat((obs_action, pred_action_gt), dim=0)
        input_goal = torch.cat((obs_goal, pred_goal_gt), dim=0)

        if training_step == 1:
            pred_goal_fake = model(
                input_traj, input_goal, seq_start_end, 1, training_step,
            )
            l2_loss_rel.append(
                utils.l2_loss(pred_goal_fake, pred_goal_gt, loss_mask, mode="raw")
            )
        elif training_step == 2:
            pred_action_fake = model(
                input_traj, input_goal, seq_start_end, 1, training_step,
            )
            l2_loss_rel.append(
                utils.l2_loss(pred_action_fake, pred_action_gt, loss_mask, mode="raw")
            )

        l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end-start)
            _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)
            _l2_loss_rel = torch.min(_l2_loss_rel) / ((pred_action_fake.shape[0]) * (end-start))
            l2_loss_sum_rel += _l2_loss_rel

        loss += l2_loss_sum_rel
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_every:
            progress.display(batch_idx)

    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, val_loader, epoch, writer):
    ADE = utils.AverageMeter("ADE", ":.6f")
    FDE = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ADE, FDE], prefix="Test: ")

    model.eval()
    with torch.no_grad:
        for i, batch in enumerate(val_loader):
            (
                obs_traj,          # obs_traj, [8,1413,2]
                pred_traj_gt,      # pred_traj, [12,1413,2]
                obs_traj_rel,      # obs_traj_rel, [8,1413,2]
                pred_traj_gt_rel,  # pred_traj_rel, [12, 1413,2]
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            loss_mask = loss_mask[:, args.obs_len:]
            obs_action = utils.cal_action(obs_traj_rel)
            obs_goal = utils.cal_goal(obs_traj_rel, obs_action)
            pred_action_gt = utils.cal_action(pred_traj_gt_rel)
            pred_goal_gt = utils.cal_goal(pred_traj_gt_rel, pred_action_gt)

            pred_action_fake = model(
                obs_action, obs_goal, seq_start_end,
            )

            # 是否还需要做处理 ?
            ADE_, FDE_ = utils.cal_ADE_FDE(pred_action_gt, pred_action_fake)
            ADE_ = ADE_ / (obs_traj.shape[1] * args.pred_len)
            FDE_ = FDE_ / (obs_traj.shape[1])
            ADE.update(ADE_, obs_traj.shape[1])
            FDE.update(FDE_, obs_traj.shape[1])

            if i % args.print_every == 0:
                progress.display(i)

        logging.info(
            "* ADE {ade.avg:.3f} FDE {fde.avg:.3f}".format(ade=ADE, fde=FDE)
        )
        writer.add_scalar("val_ade", ADE.avg, epoch)

    return ADE.avg


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
        action_encoder_input_dim=args.action_encoder_input_dim,
        action_encoder_hidden_dim=args.action_encoder_hidden_dim,
        goal_encoder_input_dim=args.goal_encoder_hidden_dim,
        goal_encoder_hidden_dim=args.goal_encoder_hidden_dim,
        goal_decoder_input_dim=args.goal_decoder_hidden_input_dim,
        goal_decoder_hidden_dim=args.goal_decoder_hidden_dim,
        action_decoder_input_dim=args.action_decoder_input_dim,
        action_decoder_hidden_dim=args.acton_decoder_hidden_dim,
        n_units=args.n_units,
        n_heads=args.n_heads,
        dropout=args.dropout,
        alpha=args.alpha,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
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


