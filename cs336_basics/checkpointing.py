import torch
import os
import typing
def save_checkpoint(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        iteration:int,
        out:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]]
):
    check_point={
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "iteration":iteration
    }
    torch.save(check_point,out)
def load_checkpoint(
        src:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]],
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer
):
    check_point=torch.load(src,map_location="cpu")
    model.load_state_dict(check_point["model_state_dict"])
    optimizer.load_state_dict(check_point["optimizer_state_dict"])
    return check_point["iteration"]
