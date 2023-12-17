from trainer import Trainer
import argparse
from PIL import Image
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='gan')
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--l1_coef", default=50, type=float)
    parser.add_argument("--l2_coef", default=100, type=float)
    parser.add_argument("--diter", default=5, type=int)
    parser.add_argument("--cls", default=False, action='store_true')
    parser.add_argument("--iter_log_file", default='iter_log.txt')
    parser.add_argument("--epoch_log_file", default='epoch_log.txt')
    parser.add_argument("--save_path", default='')
    parser.add_argument("--inference", default=False, action='store_true')
    parser.add_argument('--pre_trained_disc', default=None)
    parser.add_argument('--pre_trained_gen', default=None)
    parser.add_argument('--dataset', default='flowers')
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--local-rank', type=int, help='Local rank of the process')
    args = parser.parse_args()
    
    trainer = Trainer(type=args.type,
                      dataset=args.dataset,
                      split=args.split,
                      lr=args.lr,
                      diter=args.diter,
                      iter_log_file=args.iter_log_file,
                      epoch_log_file=args.epoch_log_file,
                      save_path=args.save_path,
                      l1_coef=args.l1_coef,
                      l2_coef=args.l2_coef,
                      pre_trained_disc=args.pre_trained_disc,
                      pre_trained_gen=args.pre_trained_gen,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      epochs=args.epochs
                      )
    
    if not args.inference:
        trainer.train(args.cls)
    else:
        trainer.predict()

if __name__ == '__main__':
    main()
