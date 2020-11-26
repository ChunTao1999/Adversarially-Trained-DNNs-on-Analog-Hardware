import pandas as pd
import matplotlib.pyplot as plt
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices = ['cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--logdir')

    args = parser.parse_args()

    load_file = 'log/{}/{}/results.txt'.format(args.dataset, args.logdir)
    df = pd.read_csv(load_file)
    fig1 = plt.figure()
    ax11 = fig1.add_subplot(1,2,1)
    ax12 = fig1.add_subplot(1,2,2)

    df[['test_loss', 'train_loss']].plot(title='LOSS: {}-{}'.format(args.dataset, args.logdir), ax = ax11)
    df[['test_error1', 'train_error1']].plot(title='Error: {}-{}'.format(args.dataset, args.logdir), ax = ax12)

    plt.show()



