import fastai.optimizer as opt
from functools import partial


def get_optimizer(run_params):
    # Scheduling
    # FixMatch is highly influenced by the Optimizer and its parameters
    # TODO: Study which parameters are best for this use-case
    if run_params["SSL"] == run_params["SSL_FIX_MATCH"]:
        # sched = {'lr': SchedCos(run_params['OPT_LR'], run_params['OPT_LR']*math.cos(7*math.pi/16))}
        # cbs.append(ParamScheduler(sched))
        # opt_params = {
        #     'moms': (run_params['MOMENTUM'],)*3, # 0.9 according to FixMatch paper
        #     'wd': run_params['OPT_WD']
        # }
        # opt_func = SGD
        opt_func = opt.Adam
        opt_params = {}
    else:

        if type(run_params["OPTIMIZER"]) is str:
            if run_params["OPTIMIZER"] == "AdaBelief":
                from adabelief_pytorch import AdaBelief

                opt_func = partial(
                    opt.OptimWrapper,
                    opt=AdaBelief,
                    betas=(run_params["OPT_MOM"], run_params["OPT_SQR_MOM"]),
                    weight_decay=run_params["OPT_WD"],
                    print_change_log=False,
                )
            elif run_params["OPTIMIZER"] == "RangerAdaBelief":
                from ranger_adabelief import RangerAdaBelief

                opt_func = partial(
                    opt.OptimWrapper,
                    opt=RangerAdaBelief,
                    betas=(run_params["OPT_MOM"], run_params["OPT_SQR_MOM"]),
                    weight_decay=run_params["OPT_WD"],
                    alpha=0.5,
                    k=6,
                    N_sma_threshhold=5,
                )
            else:
                opt_func = getattr(opt, run_params["OPTIMIZER"])
        else:
            opt_func = run_params["OPTIMIZER"]
        opt_params = {"wd": run_params["OPT_WD"]}

    return opt_func, opt_params
