import json
import numpy as np
import argparse
import pandas as pd
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import os
from tqdm import tqdm
from Targeted_attack import fgsm_loss, cw_fgsm_loss, pgd_loss, cw_pgd_loss, apgd_loss, cw_apgd_loss  #, trades_loss, cw_trades_loss, fat_loss, cw_fat_loss
from utils import dev, normalize_cifar, normalize_cifar_100, get_dataset, weight_average,load_txt, CIFAR10C
from model import PreActResNet18
from model_wrn import WRN
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import StepLR
import argparse
import glob
import numpy as np
import os
import pprint
import torch
import torchvision
import tqdm
from glob import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', default=350, type=int)
    parser.add_argument('--model', default='PRN', type=str, choices=['PRN', 'WRN']) #
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--lr_schedule', default='step3', type=str, choices=["step3", "step4", "steplr"])
    parser.add_argument('--mode', default='TRADES', type=str, choices=['AT', 'TRADES', 'FAT'])
    parser.add_argument('--attack_type', default='APGD', type=str, choices=['PGD', 'FGSM','APGD'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=int)
    parser.add_argument('--norm', default='Linf', type=str)
    parser.add_argument('--beta', default=6, type=int)  # beta for TRADES
    parser.add_argument('--tau', default=3, type=int)   #  tau for FAT
    parser.add_argument('--fname', type=str, default='auto') #TODO
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--ccm', action='store_true', default=True) # CCM
    parser.add_argument('--ccr', action='store_true') # CCR
    parser.add_argument('--random_target', action='store_true') 
    parser.add_argument('--adaptive_eps', nargs="*", choices=["G-cfps", "G-cfns", "G-rob", "T-cfps", "T-cfns", "T-rob"])
    parser.add_argument('--adaptive_eps_aggr', type=str, choices=["avg", "min"], default="avg")
    parser.add_argument('--lambda-1', default=1, type=float)
    parser.add_argument('--lambda-2', default=0.5, type=float)
    parser.add_argument('--lambda-c', default=1.5, type=float)
    parser.add_argument('--lambda-r', default=1.5, type=float)
    parser.add_argument('--kldiv', default=2.0, type=float)
    parser.add_argument('--begin', default=1, type=int)
    parser.add_argument('--decay-rate', default=0.88 ,type=float)
    parser.add_argument('--untargeted', type=int, default=False)
    parser.add_argument('--thershold', default=0.24, type=float)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_workers_train', default=8, type=int)
    parser.add_argument('--num_workers_valid', default=4, type=int)
    parser.add_argument('--num_workers_test', default=4, type=int)
    parser.add_argument('--corruptions_path', default='/home/tejaswini/Experiments_Teja/Targeted-Adversarial-Training/corruptions.txt',type=str)
    return parser.parse_args()

class CW_log():
    def __init__(self, num_classes) -> None:
        self.class_num = num_classes
        self.flip_count = 0
        self.clean = edict(
            N = 0,
            gt = edict(
                correct = 0,
                fp_by_class = np.zeros(num_classes),
                tp_by_class = np.zeros(num_classes),
                fn_by_class = np.zeros(num_classes)
            ),
            target = edict(
                correct = 0,
                fp_by_class = np.zeros(num_classes),
                tp_by_class = np.zeros(num_classes),
                fn_by_class = np.zeros(num_classes)
            )
        )
        self.robust = edict(
            N = 0,
            gt = edict(
                correct = 0,
                fp_by_class = np.zeros(num_classes),
                tp_by_class = np.zeros(num_classes),
                fn_by_class = np.zeros(num_classes)
            ),
            target = edict(
                correct = 0,
                fp_by_class = np.zeros(num_classes),
                tp_by_class = np.zeros(num_classes),
                fn_by_class = np.zeros(num_classes)
            )
        )
        self.corruptions = edict(
            N = 0,
            gt = edict(
                correct = 0,
                fp_by_class = np.zeros(num_classes),
                tp_by_class = np.zeros(num_classes),
                fn_by_class = np.zeros(num_classes)
            ),
            target = edict(
                correct = 0,
                fp_by_class = np.zeros(num_classes),
                tp_by_class = np.zeros(num_classes),
                fn_by_class = np.zeros(num_classes)
            )
        )
        
    def reset(self,which,num_classes=10):
        if which == "corruptions":
            d = getattr(self, which)
            d.N = 0
            d.gt.correct = 0
            d.gt.tp_by_class = np.zeros(num_classes)
    
    def update(self, which, output, y, y_t,flips):
        assert which in ["clean", "robust", "corruptions"]
        d = getattr(self, which)
        
        d.N += len(output)
        pred = output.max(1)[1].cpu().numpy()
        y = y.cpu().numpy()
        y_t = y_t.cpu().numpy()
        
        correct_gt = pred == y
        correct_target = pred == y_t
        
        d.gt.correct += correct_gt.sum()
        d.target.correct += correct_target.sum()
        if which == "robust":
            self.flip_count += flips
        else:
            self.flip_count = self.flip_count
        
        for i, c in enumerate(y):
            if correct_gt[i]:
                d.gt.tp_by_class[c] += 1
            else:
                d.gt.fp_by_class[pred[i]] += 1
                d.gt.fn_by_class[c] += 1
            if correct_target[i]:
                d.target.tp_by_class[c] += 1
            else:
                d.target.fp_by_class[pred[i]] += 1
                d.target.fn_by_class[c] += 1
    
    def result(self):
        # N = self.N
        # m = self.class_num
        # return (
        #     self.clean_acc / N,
        #     self.robust_acc / N,
        #     m*self.cw_clean / N,
        #     m*self.cw_robust / N,
        #     self.cw_cfps_clean / self.in_correct_clean,
        #     self.cw_cfps_robust / self.in_correct_robust
        # )
        return edict(
            clean_acc_gt = self.clean.gt.correct / self.clean.N,
            clean_acc_target = self.clean.target.correct / self.clean.N,
            robust_acc_gt = self.robust.gt.correct / self.robust.N,
            robust_acc_target = self.robust.target.correct / self.robust.N,
            clean_cw_acc_gt = self.clean.gt.tp_by_class * self.class_num / self.clean.N,
            clean_cw_acc_target = self.clean.target.tp_by_class * self.class_num / self.clean.N,
            robust_cw_acc_gt = self.robust.gt.tp_by_class * self.class_num / self.robust.N,
            robust_cw_acc_target = self.robust.target.tp_by_class * self.class_num / self.robust.N,
            clean_cw_cfps_gt = self.clean.gt.fp_by_class / (self.clean.N - self.clean.gt.correct),
            clean_cw_cfps_target = self.clean.target.fp_by_class / (self.clean.N - self.clean.target.correct),
            robust_cw_cfps_gt = self.robust.gt.fp_by_class / (self.robust.N - self.robust.gt.correct),
            robust_cw_cfps_target = self.robust.target.fp_by_class / (self.robust.N - self.robust.target.correct),
            clean_cw_cfns_gt = self.clean.gt.fn_by_class / (self.clean.N - self.clean.gt.correct),
            clean_cw_cfns_target = self.clean.target.fn_by_class / (self.clean.N - self.clean.target.correct),
            robust_cw_cfns_gt = self.robust.gt.fn_by_class / (self.robust.N - self.robust.gt.correct),
            robust_cw_cfns_target = self.robust.target.fn_by_class / (self.robust.N - self.robust.target.correct),
            flip_score = self.flip_count / (self.robust.N)
        )
    def result_corruptions(self):
        return   edict(
            acc_gt = self.corruptions.gt.correct / self.corruptions.N,
            cw_acc_gt = self.corruptions.gt.tp_by_class * self.class_num / self.corruptions.N
        )

##########
# [0] = 

def train_epoch(
        model,
        loader,
        opt,
        device,
        attack,
        eps,
        beta,
        alpha,
        n_iters
    ):
    model.train()
    logger = CW_log(num_classes=num_classes)
    # loader = tqdm(loader)
    list_gt_train = []
    list_target_train = []
    list_pred_train_clean = []
    list_pred_train_robust = []
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        if args.random_target:
            y_t = get_rand_target(y, num_classes=num_classes)
        else:
            probs = eps
            num_samples = y.shape[0]
            sample_indices = torch.multinomial(probs, num_samples, replacement=True)
            y_t = sample_indices
        loss, output = attack(
            model = model,
            normalize = normalize,
            x = x,
            y = y,
            y_t = y_t,
            cw_eps = eps,
            beta = beta,
            alpha = alpha,
            attack_mode_UT = args.untargeted,
            n_iters = 10
        )
        if args.kldiv > 0:
            with torch.no_grad():
                output_clean = model(normalize(x))
            loss_kl = torch.nn.KLDivLoss(reduction="mean")(
                torch.nn.functional.log_softmax(output, dim=1),
                torch.nn.functional.log_softmax(output_clean, dim=1)
            )
            loss = loss + 2*loss_kl
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        robust_predictions = output.max(1)[1].cpu().numpy()
        clean_output = model(normalize(x)).detach()
        clean_predictions = clean_output.max(1)[1].cpu().numpy()
        flip_count = np.sum(clean_predictions!=robust_predictions)
        logger.update("robust", output, y, y_t, flip_count)
        logger.update("clean", clean_output, y, y_t, flip_count)
        if args.debug:
            break
        list_gt_train.append(y.cpu().numpy())
        list_target_train.append(y_t.cpu().numpy())
        list_pred_train_clean.append(clean_predictions)
        list_pred_train_robust.append(robust_predictions)
    train_epoch_data = {
            "gt_train": [item for iteration in list_gt_train for item in iteration],  
            "target_train": [item for iteration in list_target_train for item in iteration],
            "pred_train_clean" : [item for iteration in list_pred_train_clean for item in iteration],
            "pred_train_robust": [item for iteration in list_pred_train_robust for item in iteration]
        }
    save_epoch_data(epoch,"train",train_epoch_data)
    return logger.result()

def eval_epoch(
        model,
        loader,
        device,
        attack,
        eps,
        beta,
        alpha,
        n_iters,
        type
    ):
    model.eval()
    logger = CW_log(num_classes=num_classes)
    # loader = tqdm(loader)
    list_gt_eval = []
    list_target_eval = []
    list_pred_eval_clean = []
    list_pred_eval_robust = []
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        #y_t = torch.randint(0,10,(y.shape[0],)).cuda()  #random target
        y_t = y
        _, output = attack(
            model = model,
            normalize = normalize,
            x = x,
            y = y,
            y_t = y_t,
            eps = eps,
            beta = beta,
            alpha = alpha,
            n_iters = n_iters
        )
        robust_predictions = output.max(1)[1].cpu().numpy()
        clean_output = model(normalize(x)).detach()
        clean_predictions = clean_output.max(1)[1].cpu().numpy()
        flip_count = np.sum(clean_predictions!=robust_predictions)
        logger.update("robust", output, y, y_t, flip_count)
        logger.update("clean", clean_output, y, y_t, flip_count)
        if args.debug:
            break
        list_gt_eval.append(y.cpu().numpy())
        list_target_eval.append(y_t.cpu().numpy())
        list_pred_eval_clean.append(clean_predictions)
        list_pred_eval_robust.append(robust_predictions)
    eval_epoch_data = {
            f"gt_{type}": [item for iteration in list_gt_eval for item in iteration],  
            f"target_{type}": [item for iteration in list_target_eval for item in iteration],
            f"pred_{type}_clean" : [item for iteration in list_pred_eval_clean for item in iteration],
            f"pred_{type}_robust": [item for iteration in list_pred_eval_robust for item in iteration]
        }
    save_epoch_data(epoch,type,eval_epoch_data)
    return logger.result()

def eval_corruptions(model):
    CORRUPTIONS = load_txt(args.corruptions_path)
    MEAN = [0.49139968, 0.48215841, 0.44653091]
    STD  = [0.24703223, 0.24348513, 0.26158784]
    model.to(device)
    model.eval()
    logger = CW_log(num_classes=num_classes)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    corruptions = CORRUPTIONS
    accs = dict()
    list_corruptions = []
    for ci, cname in enumerate(corruptions):
            data_root = 'data'
            # load dataset
            dataset = CIFAR10C(
                    os.path.join(data_root, 'CIFAR-10-C'),
                    cname, corruptions=CORRUPTIONS, transform=transform
                )
            corruption_loader = DataLoader(dataset, batch_size=1024,
                                shuffle=False, num_workers=4)
            with torch.no_grad():
                for itr, (x, y) in enumerate(corruption_loader):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, dtype=torch.int64, non_blocking=True)
                    y_t = y
                    z = model(x)
                    loss = F.cross_entropy(z, y)
                    logger.update("corruptions", z, y, y_t, flips=0)
                corruptions = {f"{cname}" : logger.result_corruptions()}
                list_corruptions.append(corruptions)
                logger.reset("corruptions")
    return list_corruptions

def get_rand_target(label, num_classes):
    target = torch.randint_like(label, 0, num_classes)
    diff = target == label
    while diff.any():
        target[diff] = torch.randint_like(target[diff], 0, num_classes)
        diff = target == label
    return target
    
def save_epoch_data(epoch,epoch_data_type, epoch_data):
    base_dir = "epoch_info"
    os.makedirs(base_dir, exist_ok=True)
    data_dir = os.path.join(base_dir, epoch_data_type)
    epoch_dir = os.path.join(data_dir,args.fname, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    filename = f"data_epoch_{epoch}.npy"
    file_path = os.path.join(epoch_dir, filename)
    np.save(file_path, epoch_data)


def lr_schedule(t):
    if t / args.epochs < 0.5:
        return args.lr_max
    elif t / args.epochs < 0.75:
        return args.lr_max / 10.
    else:
        return args.lr_max / 100.

def lr_schedule_new(t):
    if t / args.epochs < 0.25:
        return args.lr_max
    elif t / args.epochs < 0.5:
        return args.lr_max / 10.
    elif t / args.epochs < 0.75:
        return args.lr_max / 100.
    else:
        return args.lr_max / 1000.

def lr_schedule_wrn(t):
    if t < 75:
        return args.lr_max
    elif t < 90:
        return args.lr_max / 10.
    else:
        return args.lr_max / 100.
    
if __name__ == '__main__':
    args = get_args()
    ################
    # args.mode = "AT"
    # args.ccm = True
    # args.random_target = True
    # args.adaptive_eps = None
    # args.lambda_1 = 0.5
    # args.untargeted = 1
    ################
    num_classes = 10 if args.data=="cifar10" else 100
    normalize = normalize_cifar if args.data=="cifar10" else normalize_cifar_100
    print(f"num_classes = {num_classes}")
    print(f"normalize = {normalize}")
    if args.fname == 'auto':
        args.fname = "_".join([
            args.data,
            args.model,
            args.mode,
            args.attack_type,
            str(args.untargeted),
            f"kldiv={args.kldiv}",
            f"ccm={args.ccm}",
            f"ccr={args.ccr}",
            f"rand={args.random_target}",
            "adapt=" + ",".join(args.adaptive_eps) if args.adaptive_eps else "None",
            f"alpha={args.pgd_alpha}",
            f"its={args.attack_iters}",
            f"lr={args.lr_max}",
            f"schedule={args.lr_schedule}",
            f"bs={args.batch_size}"
        ])
    if args.prefix:
        args.fname = args.prefix + "_" + args.fname
    print(args)
    fname = args.fname
    device = dev(args.device)
    eps = args.epsilon / 255.       # 8/255
    alpha = args.pgd_alpha / 255.   # 2/255
    beta = args.beta / 1.           # 6
    class_eps = torch.ones(num_classes).to(device) * eps
    class_beta = torch.ones(num_classes).to(device) * (beta/(1+beta))
    iteration = args.attack_iters  # 10
    epochs = args.epochs if args.model == 'PRN' else 100    # 200 epochs
    train_loader, valid_loader, test_loader = get_dataset(
        dataset = args.data,
        num_workers_train = args.num_workers_train,
        num_workers_valid = args.num_workers_valid,
        num_workers_test = args.num_workers_test,
        batch_size_train = args.batch_size,
        batch_size_valid = args.batch_size,
        batch_size_test = args.batch_size,
    )
    print(train_loader.dataset)
    print(valid_loader.dataset)
    print(test_loader.dataset)

    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('logs'):
        os.mkdir('logs')
        
    if not os.path.exists('models/'+args.fname):
        os.mkdir('models/'+args.fname)
    if not os.path.exists('logs/'+args.fname):
        os.mkdir('logs/'+args.fname)
    with open(f'logs/{fname}/config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    if args.model == 'PRN':
        model = PreActResNet18(num_classes=num_classes).to(device)

    elif args.model == 'WRN':
        model = WRN().to(device)
    else:
        raise ValueError
    
    # init weight averaged model
    EMA_model = PreActResNet18(num_classes=num_classes).to(device) if args.model == 'PRN' else WRN().to(device)

    FAWA_model = PreActResNet18(num_classes=num_classes).to(device) if args.model == 'PRN' else WRN().to(device)

    EMA_model.eval()
    FAWA_model.eval()
    # print(EMA_model, FAWA_model)  both models are PreActResNet18
    SEAT_init = False
    
    params = model.parameters()
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    if args.lr_schedule == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.96)
    else:
        scheduler = None
    log_data = [] # Epochs * 7:    Epoch, train_clean, train, valid_clean, valid, test_clean, test
    cw_data = []  # Epochs * 6 * 10:    Epoch, min-{train_clean, train, valid_clean, valid, test_clean, test}
    EMA_log, FAWA_log = [], []
    
    log_train_results = []
    log_valid_results = []
    log_test_results = []
    log_EMA_results = []
    log_FAWA_results = []
    log_class_wise_eps = []
    
    save_threshold = [0, 0, 0] # robust+min_robust, for main, EMA, FAWA
    for epoch in tqdm(range(epochs), desc="Epoch"):
        # update learning rate
        if args.model == 'WRN':
            lr = lr_schedule_wrn(epoch)
        else:
            if args.lr_schedule == "step3":
                lr = lr_schedule(epoch)
            elif args.lr_schedule == "step4":
                lr = lr_schedule_new(epoch)
            elif args.lr_schedule == "steplr":
                if epoch > 0:
                    scheduler.step()
                lr = None
            else:
                raise RuntimeError(f"lr_schedule not understood: '{args.lr_schedule}'.")
        if lr is not None:
            opt.param_groups[0].update(lr=lr)
        
        # train
        model.train()
        # ccm
        if epoch >= args.begin:
            if args.adaptive_eps:
                class_eps = []
                # how to scale eps?
                for _adaptive_eps in args.adaptive_eps:
                    if _adaptive_eps == "G-rob":
                        scaling = log_train_results[-1].robust_cw_acc_gt
                    elif _adaptive_eps == "T-rob":
                        scaling = log_train_results[-1].robust_cw_acc_target
                    elif _adaptive_eps == "G-cfps":
                        scaling = log_train_results[-1].robust_cw_cfps_gt
                    elif _adaptive_eps == "T-cfps":
                        scaling = log_train_results[-1].robust_cw_cfps_target
                    elif _adaptive_eps == "G-cfns":
                        scaling = log_train_results[-1].robust_cw_cfns_gt
                    elif _adaptive_eps == "T-cfns":
                        scaling = log_train_results[-1].robust_cw_cfns_target
                    else:
                        raise NotImplementedError("shouldn't get here")
                    if "-rob" in _adaptive_eps:
                        lambd = args.lambda_r
                    else:
                        lambd = args.lambda_c
                    class_eps += [(np.ones(num_classes) * lambd + scaling) * eps]
                # combine eps
                class_eps = np.stack(class_eps)
                if args.adaptive_eps_aggr == "avg":
                    class_eps = class_eps.mean(axis=0)
                elif args.adaptive_eps_aggr == "min":
                    class_eps = class_eps.min(axis=0)
                else:
                    raise NotImplementedError("shouldn't get here")
                
                # train_robust = torch.tensor(train_robust).to(device)
                # if args.cfps:
                #     if args.gt_targets:
                #         train_robust = log_train_results[-1].robust_cw_cfps_gt
                #     else:
                #         train_robust = log_train_results[-1].robust_cw_cfps_target
                # else:
                #     if args.gt_targets:
                #         train_robust = log_train_results[-1].robust_cw_acc_gt
                #     else:
                #         train_robust = log_train_results[-1].robust_cw_acc_target
                class_eps = torch.tensor(class_eps).to(device)
            else:
                class_eps = torch.ones(num_classes).to(device) * eps
        
        # ccr
        if args.ccr and epoch >= args.begin:
            for i in range(num_classes):
                class_beta[i] = (args.lambda_2+train_robust[i]) * beta / (1 + (args.lambda_2+train_robust[i])*beta)
        else:
            class_beta = torch.ones(num_classes).to(device) * (beta/(1+beta))
            # going class_beta

        # set tau for FAT
        if args.mode == 'FAT':
            class_beta = args.tau
        
        if args.mode == 'AT':
            if args.ccm:
                if args.attack_type == 'FGSM':
                    attack = cw_fgsm_loss
                elif args.attack_type == 'PGD':
                    attack = cw_pgd_loss
                elif args.attack_type == 'APGD':
                    attack = cw_apgd_loss
            else:
                if args.attack_type == 'FGSM':
                    attack = fgsm_loss
                elif args.attack_type == 'PGD':
                    attack = pgd_loss
                elif args.attack_type == 'APGD':
                    attack = apgd_loss
                    
        elif args.mode == 'TRADES':
            if args.ccm:
                attack = cw_trades_loss
            else:
                attack = trades_loss
        elif args.mode == 'FAT':
            if args.ccm:
                attack = cw_fat_loss
            else:
                attack = fat_loss


        if args.ccm:
            train_result = train_epoch(
                model = model,
                loader = train_loader,
                opt = opt,
                device = device,
                attack = attack,
                eps = class_eps,
                beta = class_beta,
                alpha = alpha,
                n_iters = iteration
            )
        else:
            train_result = train_epoch(
                model = model,
                loader = train_loader,
                opt = opt,
                device = device,
                attack = attack,
                eps = eps,
                beta = class_beta,
                alpha = alpha,
                n_iters = iteration
            )
        print()
        print("########## Training Result ##########")
        for k, v in train_result.items():
            print(k)
            print(v)
        
        model.eval()
        # test
        test_result = eval_epoch(
            model = model,
            loader = test_loader,
            device = device,
            attack = apgd_loss,
            eps = 8./255.,
            beta = beta,
            alpha = 2./255.,
            n_iters = 10,
            type = "test"
        )
        print()
        print("########## Test Result ##########")
        for k, v in test_result.items():
            print(k)
            print(v)
    
        # valid
        valid_result = eval_epoch(
            model = model,
            loader = valid_loader,
            device = device,
            attack = apgd_loss,
            eps = 8./255.,
            beta = beta,
            alpha = 2./255.,
            n_iters = 10,
            type = "valid"
        )
        print()
        print("########## Valid Result ##########")
        for k, v in valid_result.items():
            print(k)
            print(v)
        # weight average
        # EMA
        weight_average(EMA_model, model, args.decay_rate, epoch==0)
        EMA_result = eval_epoch(
            model = EMA_model,
            loader = test_loader,
            device = device,
            attack = apgd_loss,
            eps = 8./255.,
            beta = beta,
            alpha = 2./255.,
            n_iters = 10,
            type = "test_EMA"
        )
        print()
        print("########## EMA Result ##########")
        for k, v in EMA_result.items():
            print(k)
            print(v)
        
        # FAWA
        R_min = valid_result.robust_cw_acc_gt.min()
        if R_min >= args.thershold:
            if not SEAT_init:
                SEAT_init = True
                weight_average(FAWA_model, model, args.decay_rate, True)
            else:
                weight_average(FAWA_model, model, args.decay_rate, False)
        else:
            weight_average(FAWA_model, model, 1., False)
        FAWA_result = eval_epoch(
            model = FAWA_model,
            loader = test_loader,
            device = device,
            attack = apgd_loss,
            eps = 8./255.,
            beta = beta,
            alpha = 2./255.,
            n_iters = 10,
            type = "test_FAWA"
        )
        print()
        print("########## FAWA Result ##########")
        for k, v in FAWA_result.items():
            print(k)
            print(v)

        corruption_result =  eval_corruptions(model)
        print()
        print("########## Corruption Result ##########")
        for i in corruption_result:
            print(i)
        
        # log result
        log_train_results += [train_result]
        log_valid_results += [valid_result]
        log_test_results += [test_result]
        log_EMA_results += [EMA_result]
        log_FAWA_results += [FAWA_result]
        
        for _log, fp in [
                (log_train_results, "log_train_results"),
                (log_valid_results, "log_valid_results"),
                (log_test_results, "log_test_results"),
                (log_EMA_results, "log_EMA_results"),
                (log_FAWA_results, "log_FAWA_results")
            ]:
            torch.save(_log, f"logs/{args.fname}/{fp}.pth")
        
        # plot
        log_class_wise_eps += [class_eps.cpu().numpy()]
        df = pd.DataFrame(np.stack(log_class_wise_eps))
        df.to_csv(f'logs/{args.fname}/class_wise_eps.csv')
        
        print()
        print("########## Class-wise eps *255 ##########")
        print(log_class_wise_eps[-1]*255)
        
        # save models
        if epoch >= 0.5 * args.epochs:
            # Main
            index = test_result.robust_acc_gt + test_result.robust_cw_acc_gt.min()
            if index >= save_threshold[0] - 0.02 or epoch >= args.epochs-5:
                torch.save(model.state_dict(), f'models/{args.fname}/{epoch}.pth')
                save_threshold[0] = max(save_threshold[0], index)
            
            # EMA
            index = EMA_result.robust_acc_gt + EMA_result.robust_cw_acc_gt.min()
            if index >= save_threshold[1] - 0.02 or epoch >= args.epochs-5:
                torch.save(EMA_model.state_dict(), f'models/{args.fname}/EMA_{epoch}.pth')
                save_threshold[1] = max(save_threshold[1], index)

            # FAMA
            index = FAWA_result.robust_acc_gt + FAWA_result.robust_cw_acc_gt.min()
            if index >= save_threshold[2] - 0.02 or epoch >= args.epochs-5:
                torch.save(FAWA_model.state_dict(), f'models/{args.fname}/FAWA_{epoch}.pth')
                save_threshold[2] = max(save_threshold[2], index)
