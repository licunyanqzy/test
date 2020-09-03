from data.loader import data_loader
import utils
import os
import argparse


# a test of labeling action and goal
parser = argparse.ArgumentParser()
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--batch_size", default=64, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
train_path = utils.get_dset_path("zara2", "train")
val_path = utils.get_dset_path("zara2", "test")
train_dset, train_loader = data_loader(args, train_path)
_, val_loader = data_loader(args, val_path)

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

    obs_action = utils.cal_action(obs_traj_rel)
    obs_goal = utils.cal_goal(obs_traj_rel, obs_action)
    pred_action_gt = utils.cal_action(pred_traj_gt_rel)
    pred_goal_gt = utils.cal_goal(pred_traj_gt_rel, pred_action_gt)

