import argparse
import os
import torch
import numpy as np
import wandb  # 建议安装：pip install wandb
from  cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import Adamw,clip_gradient_norm
from cs336_basics.scheduler import get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.losses import cross_entropy

def main():
    parser=argparse.ArgumentParser()
    #model
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--context_length",type=int,default=256)
    parser.add_argument("--d_model",type=int,default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)
    #optimizer
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--max_norm", type=float, default=1.0)
     # --- 路径与系统 ---
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--eval_batches", type=int, default=10, help="Number of batches averaged per validation")

    # --- WandB 设置 ---
    parser.add_argument("--wandb_project", type=str, default="cs336-pretraining")
    parser.add_argument("--run_name", type=str, default=None, help="WandB 实验名称")
    args=parser.parse_args()
    if args.eval_batches < 1:
        parser.error("--eval_batches must be at least 1")
    os.makedirs(args.out_dir,exist_ok=True)

    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Training data not found at {args.valid_data_path}")
    train_data=np.memmap(args.train_data_path,dtype=np.uint16,mode="r")
    valid_data=np.memmap(args.valid_data_path,dtype=np.uint16,mode="r")
    print(f"训练集大小:{len(train_data)}tokens")
    print(f"验证集大小:{len(valid_data)}tokens")



    model=TransformerLM(
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        d_model=args.d_model,
        rope_theta=actual_rope_theta,
        device=args.device,
    ).to(args.device)
    optimizer=Adamw(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=0.1
    )
    start_iter=0
    ckpt_path=os.path.join(args.out_dir,"ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter=load_checkpoint(ckpt_path,model,optimizer)
        print(f"Resuming from iteration{start_iter}")
      # 6. 初始化 WandB 监控
    wandb.init(
        project=args.wandb_project,
        name=args.run_name, 
        config=args
    )
    for it in range(start_iter,args.max_iters):
        lr=get_lr_cosine_schedule(it,args.lr,args.min_lr,args.warmup_iters,args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"]=lr
        model.train()
        x,y=get_batch(train_data,args.batch_size,args.context_length,args.device)

        logits=model(x)
        loss=cross_entropy(logits,y)

        optimizer.zero_grad()
        loss.backward()

        clip_gradient_norm(model.parameters(),args.max_norm)

        optimizer.step()

        if it %100 ==0 or it==args.max_iters-1:
            model.eval()
            with torch.no_grad():
                # Preserve training RNG; use the same validation samples each time.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(42)
                    validation_losses = []
                    for _ in range(args.eval_batches):
                        vx,vy=get_batch(valid_data,args.batch_size,args.context_length,args.device)
                        v_logits=model(vx)
                        validation_losses.append(cross_entropy(v_logits,vy).item())
                    v_loss=torch.tensor(sum(validation_losses)/len(validation_losses))
                print(f"Iter {it}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f},lr {lr:.2e}")
                wandb.log({
                    "train/loss":loss.item(),
                    "val/loss":v_loss.item(),
                    "lr":lr,
                    "iter":it+1
                })
        if it %1000 ==0 and it>0:
            save_checkpoint(model,optimizer,it+1,ckpt_path)
    save_checkpoint(model,optimizer,args.max_iters,os.path.join(args.out_dir,"ckpt_final.pt"))
    wandb.finish()
if __name__ =="__main__":
    main()

    
        
    
        
    
