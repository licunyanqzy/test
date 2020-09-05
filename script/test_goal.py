from data.loader import data_loader
import utils
import os
import argparse


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


def main(args):
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


if __name__ == '__main__':
    args = parser.parse_args()
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    checkpoint_dir = "./checkpoint"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)