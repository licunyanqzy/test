import logging
import torch
import shutil
import os


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "model_best.pth.tar")


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    # _dir = _dir.split("/")[:-1]
    # _dir = "/".join(_dir)
    return os.path.join(_dir, "datasets", dset_name, dset_type)


def average_displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode="sum"):
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.premute(1, 0, 2)
    loss = loss ** 2

    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)

    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode="sum"):
    loss = pred_pos_gt - pred_pos
    loss = loss ** 2

    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))

    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)


def cal_ADE_FDE(pred_traj_gt, pred_traj, consider_ped=None, mode="sum"):
    ADE = average_displacement_error(pred_traj, pred_traj_gt, consider_ped, mode)
    FDE = final_displacement_error(pred_traj[-1], pred_traj_gt[-1], consider_ped, mode)
    return ADE, FDE


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode="average"):
    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2

    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)
