import argparse
import os
import torch

from data.loader import data_loader
from models import TrajectoryPrediction
import utils
from utils import int_tuple


parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", default="./")
parser.add_argument("--dataset_name", default="zara2", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--dropout", default=0, type=float) # dropout rate
parser.add_argument("--alpha", default=0.2, type=float)   # alpha for the leaky relu
parser.add_argument("--action_encoder_input_dim", default=8, type=int)
parser.add_argument("--action_encoder_hidden_dim", default=32, type=int)
parser.add_argument("--goal_encoder_input_dim", default=16, type=int)
parser.add_argument("--goal_encoder_hidden_dim", default=32, type=int)
parser.add_argument("--goal_decoder_input_dim", default=16, type=int)
parser.add_argument("--goal_decoder_hidden_dim", default=32, type=int)
parser.add_argument("--action_decoder_input_dim", default=16, type=int)
parser.add_argument("--action_decoder_hidden_dim", default=32, type=int)
parser.add_argument("--dset_type", default="test", type=str)
parser.add_argument(
    "--resume", default="./model_best.pth.tar", type=str,
    metavar="PATH", help="path to latest checkpoint (default: none)"
)
parser.add_argument("--num_samples", default=20, type=int)


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def get_model(checkpoint):

    model = TrajectoryPrediction(
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
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def evaluate(args, loader, model):

    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,  # obs_traj, [8,1413,2]
                pred_traj_gt,  # pred_traj, [12,1413,2]
                obs_traj_rel,  # obs_traj_rel, [8,1413,2]
                pred_traj_gt_rel,  # pred_traj_rel, [12, 1413,2]
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            ADE, FDE = [], []

            for _ in range(args.num_samples):


    return ADE, FDE


def main(args):
    checkpoint = torch.load(args.resume)
    model = get_model(checkpoint)
    path = utils.get_dset_path(args.dataset_name, args.dset_type)
    _, loader = data_loader(args, path)
    ADE, FDE = evaluate(args, loader, model)
    print(
        "Dataset: {}, Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            args.dataset_name, args.pred_len, ADE, FDE
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(64)   # 作用 ?
    torch.backends.cudnn.deterministic = True   # 作用 ?
    torch.backends.cudnn.benchmark = False   # 作用 ?
    main(args)
