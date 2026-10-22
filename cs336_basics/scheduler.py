import math
def get_lr_cosine_schedule(
        it:int,
        max_learning_rate:float,
        min_learning_rate:float,
        warmup_iters:int,
        cosine_cycle_iters:int
        ):
    if it<warmup_iters:
        return max_learning_rate*it/warmup_iters
    if it>cosine_cycle_iters:
        return min_learning_rate
    else :
        omiga=math.pi/(cosine_cycle_iters-warmup_iters)
        return 0.5*(max_learning_rate-min_learning_rate)*(1+math.cos((it-warmup_iters)*omiga))+min_learning_rate
