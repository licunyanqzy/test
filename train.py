import argparse
import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import line_profiler
import torchprof

from data.loader import data_loader
from models import TrajectoryPrediction
import utils
from utils import int_tuple


parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
parser.add_argument("--dataset_name", default="zara2", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=0, type=int)    # 4 -> 8 or 16 !
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--batch_size", default=64, type=int)       # 64 -> 128 ! ?
parser.add_argument("--seed", default=72, type=int)

parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)

parser.add_argument("--action_encoder_hidden_dim", default=32, type=int)
parser.add_argument("--action_input_dim", default=16, type=int)
parser.add_argument("--goal_input_dim", default=16, type=int)
parser.add_argument("--goal_encoder_hidden_dim", default=32, type=int)
parser.add_argument("--distance_embedding_dim", default=32, type=int)

parser.add_argument("--hidden_units", default="16", type=str)   # n_units
parser.add_argument("--heads", default="4,1", type=str)   # n_heads
parser.add_argument("--dropout", default=0, type=float)     # dropout rate
parser.add_argument("--alpha", default=0.2, type=float)   # alpha for the leaky relu

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian", type=str)

parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--num_epoch", default=400, type=int)

parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument("--use_gpu", default=True, type=bool)
parser.add_argument("--print_every", default=10, type=int)

parser.add_argument("--resume", default="", type=str)


bestADE = 100


def train(args, model, sigma, train_loader, optimizer, epoch, writer):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,           # obs_traj, [8,1413,2]
            pred_traj_gt,       # pred_traj, [12,1413,2]
            obs_traj_rel,       # obs_traj_rel, [8,1413,2]
            pred_traj_gt_rel,   # pred_traj_rel, [12, 1413,2]
            non_linear_ped,
            loss_mask,          # [1413, 20]
            seq_start_end,      # [64, 2]
        ) = batch

        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len:]

        input_traj = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)     # [20,1413,2]
        input_goal = utils.cal_goal(input_traj)     # [20,1413,2]
        pred_goal_gt_rel = input_goal[args.obs_len:]     # [12,1413,2]

        pred_goal_fake, pred_action_fake = model(       # [12,1413,2] [12,1413,2]
            input_traj, input_goal, seq_start_end, 1    # teacher_forcing_ratio 的取值 ?
        )

        # 输入/输出 traj
        pred_traj_fake = pred_action_fake

        l2_traj = utils.l2_loss(pred_traj_fake, pred_traj_gt_rel, loss_mask, mode="raw").unsqueeze(1)
        l2_goal = utils.l2_loss(pred_goal_fake, pred_goal_gt_rel, loss_mask, mode="raw").unsqueeze(1)
        l2_traj_sum = torch.zeros(1).to(pred_traj_gt)
        l2_goal_sum = torch.zeros(1).to(pred_traj_gt)
        for start, end in seq_start_end.data:
            _l2_traj = torch.narrow(l2_traj, 0, start, end-start)
            _l2_goal = torch.narrow(l2_goal, 0, start, end-start)
            _l2_traj = torch.sum(_l2_traj, dim=0)
            _l2_goal = torch.sum(_l2_goal, dim=0)
            _l2_traj = torch.min(_l2_traj) / (pred_traj_fake.shape[0] * (end-start))
            _l2_goal = torch.min(_l2_goal) / (pred_goal_fake.shape[0] * (end-start))
            l2_traj_sum += _l2_traj
            l2_goal_sum += _l2_goal

        l2_weight = 0.5 / (sigma[0] ** 2) * l2_traj_sum + 0.5 / (sigma[1] ** 2) * l2_goal_sum \
            + torch.log(sigma[0]) + torch.log(sigma[1])

        print(l2_goal_sum)
        print(l2_traj_sum)
        print(l2_weight)

        loss += l2_weight
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()
        optimizer.step()

        # if batch_idx % args.print_every:
        progress.display(batch_idx)

    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, weightLoss, val_loader, epoch, writer):
    ADE = utils.AverageMeter("ADE", ":.6f")
    FDE = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ADE, FDE], prefix="Test: ")

    model.eval()
    weightLoss.eval()
    with torch.no_grad:
        for i, batch in enumerate(val_loader):
            (
                obs_traj,          # obs_traj, [8,235,2]
                pred_traj_gt,      # pred_traj, [12,235,2]
                obs_traj_rel,      # obs_traj_rel, [8,235,2]
                pred_traj_gt_rel,  # pred_traj_rel, [12, 235,2]
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            loss_mask = loss_mask[:, args.obs_len:]
            obs_goal = utils.cal_goal(obs_traj_rel)
            pred_goal_gt = utils.cal_goal(pred_traj_gt_rel)

            pred_goal_fake, pred_action_fake = model(   # 默认 teacher_forcing_ratio = 0.5, training_step = 2, 前者是否需要调整 ?
                obs_traj_rel, seq_start_end,  # 暂时输入/输出 traj ?
            )

            # 输入/输出 traj
            pred_traj_fake = pred_action_fake
            # pred_traj_fake = utils.action2traj(pred_action_fake)

            # 是否还需要这两步的处理 ?
            pred_traj_fake_predpart = pred_traj_fake[-args.pred_len:]
            pred_traj_fake_abs = utils.relative_to_abs(pred_traj_fake_predpart, obs_traj[-1])

            ADE_, FDE_ = utils.cal_ADE_FDE(pred_traj_gt, pred_traj_fake_abs)
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = utils.get_dset_path(args.dataset_name, "train")
    val_path = utils.get_dset_path(args.dataset_name, "test")

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    writer = SummaryWriter()

    n_units = [
        [args.action_encoder_hidden_dim]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.action_encoder_hidden_dim]
    ]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    model = TrajectoryPrediction(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        action_input_dim=args.action_input_dim,
        action_encoder_hidden_dim=args.action_encoder_hidden_dim,
        goal_input_dim=args.goal_input_dim,
        goal_encoder_hidden_dim=args.goal_encoder_hidden_dim,
        distance_embedding_dim=args.distance_embedding_dim,
        n_units=n_units,
        n_heads=n_heads,
        dropout=args.dropout,
        alpha=args.alpha,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model.cuda()
    sigma1 = torch.ones((1,), requires_grad=True)
    sigma2 = torch.ones((1,), requires_grad=True)
    params = ([p for p in model.parameters()] + [sigma1] + [sigma2])
    optimizer = optim.Adam(params, lr=args.lr)     # 是否需要为每个模块单独设置参数 ?

    if args.resume:     # start from checkpoint
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found as '{}'".format(args.resume))

    global bestADE

    for epoch in range(args.start_epoch, args.num_epoch):
        sigma = [sigma1.cuda(), sigma2.cuda()]
        train(args, model, sigma, train_loader, optimizer, epoch, writer)

        ADE = validate(args, model, val_loader, epoch, writer)
        is_best = ADE > bestADE
        bestADE = min(ADE, bestADE)

        utils.save_checkpoint(  # if ADE > bestADE, save checkpoint
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_ADE": bestADE,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
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


