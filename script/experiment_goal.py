from data.loader import data_loader
import utils
import os
import argparse
import torch
import numpy as np
import random
import logging
from torch.utils.tensorboard import SummaryWriter
from models import GoalEncoder
import torch.optim as optim


# a test of labeling action and goal
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
parser.add_argument("--dataset_name", default="zara2", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)    # The number of background threads to use for data loading
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--seed", default=72, type=int)

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


def train(args, model, train_loader, optimizer, epoch, writer):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )

    model.train()

    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,  # obs_traj, [8,1413,2]
            pred_traj_gt,  # pred_traj, [12,1413,2]
            obs_traj_rel,  # obs_traj_rel, [8,1413,2]
            pred_traj_gt_rel,  # pred_traj_rel, [12, 1413,2]
            non_linear_ped,
            loss_mask,  # [1413, 20]
            seq_start_end,  # [64, 2]
        ) = batch

        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, -1]

        pred_goal = model(obs_traj_rel)
        goal_gt = pred_traj_gt_rel[-1, :, :]
        l2_loss_rel.append(
            utils.l2_loss(pred_goal, goal_gt, loss_mask, mode="raw")
        )


        # ?
        l2_loss_sum_rel = torch.zeros(1).to(goal_gt)
        l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
            _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
            _l2_loss_rel = torch.min(_l2_loss_rel) / (
                    (pred_goal.shape[0]) * (end - start)
            )
            l2_loss_sum_rel += _l2_loss_rel

        loss += l2_loss_sum_rel
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, val_loader, epoch, writer):
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch




def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_path = utils.get_dset_path("zara2", "train")
    val_path = utils.get_dset_path("zara2", "test")
    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    writer = SummaryWriter()

    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,  # shape[8, 1413, 2], obs_traj
            pred_traj_gt,  # shape[12, 1413, 2], pred_traj
            obs_traj_rel,  # shape[8, 1413, 2], obs_traj_rel
            pred_traj_gt_rel,  # shape[12, 1413, 2], pred_traj_rel
            non_linear_ped,  # 0
            loss_mask,  # shape[1413, 20]
            seq_start_end,  # shape[64, 2]
        ) = batch

        model = GoalEncoder(
            obs_len=args.obs_len,
            goal_encoder_hidden_dim=args.goal_encoder_hidden_state,
            goal_encoder_input_dim=args.goal_encoder_input_state
        )
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.start_epoch, args.num_epoch):
            train(args, model, train_loader, optimizer, epoch, writer)


        # obs_action = utils.cal_action(obs_traj_rel)
        # obs_goal = utils.cal_goal(obs_traj_rel, obs_action)
        # pred_action_gt = utils.cal_action(pred_traj_gt_rel)
        # pred_goal_gt = utils.cal_goal(pred_traj_gt_rel, pred_action_gt)


if __name__ == '__main__':
    """
    输入轨迹序列, 输出目的地点
    测试准确度, 测试输入序列加长后准确度是否提升
    """

    args = parser.parse_args()
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    checkpoint_dir = "./checkpoint"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)
