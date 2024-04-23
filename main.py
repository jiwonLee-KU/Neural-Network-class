from argument import arg
from data_load import data_loader
from load_net import build_net
from set_net import set_net
from utils import control_random, timeit, createFolder
import os
import argparse
import pandas as pd
test_accs = []
# @timeit
def main():
    args = arg()
    # print("sub_num: ", i)
    global test_accs
    test_acc_list = []
    best_test_acc_list= []
    number_sub = 16
    for i in range(1, number_sub):
        args.train_subject = [int(i)]

        # Set save_path

        assert args.stamp is not None, "You Should enter stamp."
        if args.train_cont_path:
            args.save_path = os.path.dirname(os.path.dirname(args.train_cont_path))
        else:
            args.save_path = f"./result/{args.stamp}/{args.train_subject[0]}"
            createFolder(args.save_path)

        # seed control
        if args.seed:
            control_random(args)

        # load train / test dataset
        train_loader, val_loader, test_loader = data_loader(args)

        # import backbone model
        net = build_net(args, train_loader.dataset.X.shape)

        # make solver (runner)
        solver = set_net(args, net, train_loader, val_loader, test_loader)


        # train
        solver.experiment()

        test_acc_list.append(args.acc)

    df = pd.DataFrame(
        {'sub': range(number_sub-1), 'test_acc': test_acc_list})

    df.to_excel(f"{args.save_path}/result.xlsx", index=False)





if __name__ == '__main__':
    main()

