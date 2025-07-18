from diffusionmodel import *
from data_util3 import load_data
from data_loader import *
import numpy as np
import argparse
import torch
import time
import traceback
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def parse_args():
    config_args = {
        'lr': 0.0005,
        'dropout_gat': 0.3,
        'dropout': 0.5,
        'cuda': 0,
        'epochs_gat': 3000,
        'epochs': 500,
        'weight_decay_gat': 1e-5,
        'weight_decay': 0,
        'seed': 1314,
        'model': 'IMF_kl',
        'num-layers': 3,
        'dim': 512,
        'r_dim': 512,
        'set_channels': [64, 128, 256],
        'part_dim': 16,
        'k_w': 10,
        'k_h': 20,
        'n_heads': 2,
        'dataset': 'FB15K-237',
        'pre_trained': 1,
        'encoder': 0,
        'image_features': 1,
        'text_features': 1,
        'patience': 10,
        'eval_freq': 5,
        'lr_reduce_freq': 500,
        'gamma': 0.999,
        'bias': 1,
        'neg_num': 500,
        'neg_num_gat': 2,
        'alpha': 0.3,
        'alpha_gat': 0.2,
        'out_channels': 32,
        'kernel_size': 3,
        'batch_size': 512,
        'batch_size2': 1024,
        'save': 1
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", action="append", default=val)
    args = parser.parse_args()
    return args

args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, img_features, text_features, train_data, val_data, test_data = load_data(args.dataset)
print("Training data {:04d}".format(len(train_data[0])))

if args.model in ['ConvE', 'TuckER', 'Mutan', 'IMF', 'IMF_kl', 'IMF_conv']:
    corpus = ConvECorpus(args, train_data, val_data, test_data, entity2id, relation2id)
else:
    corpus = ConvKBCorpus(args, train_data, val_data, test_data, entity2id, relation2id)
if args.image_features:
    args.img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)
args.entity2id = entity2id
args.relation2id = relation2id

model_name = {
    'MEOW': MEOW,
}


def train_encoder(args):
    model = GAT(args)
    print(str(model))
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=float(args.gamma))
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)

    # Train Model
    t_total = time.time()
    corpus.batch_size = len(corpus.train_triples)
    corpus.neg_num = 2

    for epoch in range(args.epochs_gat):
        model.train()
        t = time.time()
        np.random.shuffle(corpus.train_triples)
        train_indices, train_values = corpus.get_batch(0)
        train_indices = torch.LongTensor(train_indices)
        if args.cuda is not None and int(args.cuda) >= 0:
            train_indices = train_indices.to(args.device)

        optimizer.zero_grad()
        entity_embed, relation_embed = model.forward(corpus.train_adj_matrix, train_indices)
        #loss = model.loss_func(train_indices, entity_embed, relation_embed)
        try:

            # 可疑的代码块
            loss = model.loss_func(train_indices, entity_embed, relation_embed)

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print("Epoch {} , epoch_time {}".format(epoch, time.time() - t))

    print("GAT training finished! Total time is {}".format(time.time()-t_total))
    if args.save:
        save_path = f'./checkpoint/{args.dataset}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 保存模型
        torch.save(model.state_dict(), f'{save_path}GAT_final.pth')
        print(f"Final model saved！")






def train_decoder(args):
    lr_options = np.linspace(0.0004, 0.0004, 1)
    dropout_options = [0.4]
    weight_options = [0.2]
    results = {}
    for lr in lr_options:
        for dropout in dropout_options:
            for weight in weight_options:
                config_key = f"lr={lr}, dropout={dropout}, weight={weight}"
                print(f'正在训练，学习率={lr}, dropout={dropout}, 权重系数={weight}')
                args.lr = lr
                args.dropout = dropout
                if args.encoder:
                    model_gat = GAT(args)
                model = model_name[args.model](args)
                print(str(model))
                optimizer = torch.optim.Adam(
                    params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
                tot_params = sum([np.prod(p.size()) for p in model.parameters()])
                print(f'Total number of parameters: {tot_params}')
                if args.cuda is not None and int(args.cuda) >= 0:
                    if args.encoder:
                        model_gat = model_gat.to(args.device)
                        model_gat.load_state_dict(
                            torch.load('./checkpoint/{}/GAT_final.pth'.format(args.dataset)), strict=False)
                        pickle.dump(model_gat.final_entity_embeddings.detach().cpu().numpy(),
                                    open(args.dataset + '/gat_entity_vec.pkl', 'wb'))
                        pickle.dump(model_gat.final_relation_embeddings.detach().cpu().numpy(),
                                    open(args.dataset + '/gat_relation_vec.pkl', 'wb'))
                    model = model.to(args.device)

                # Train Model
                t_total = time.time()
                counter = 0
                best_val_metrics = model.init_metric_dict()
                best_test_metrics = model.init_metric_dict()
                corpus.batch_size = args.batch_size
                corpus.neg_num = args.neg_num

                for epoch in range(args.epochs):
                    try:
                        model.train()
                        epoch_loss = []
                        t = time.time()
                        corpus.shuffle()
                        for batch_num in range(corpus.max_batch_num):
                            optimizer.zero_grad()
                            train_indices, train_values = corpus.get_batch(batch_num)
                            train_indices = torch.LongTensor(train_indices).to(args.device)
                            train_values = train_values.to(args.device)
                            output = model.forward(train_indices)
                            loss = model.loss_func(output, train_values, weight)  # 将权重传递到损失函数
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)# 在梯度裁剪后执行优化步骤
                            optimizer.step()
                            epoch_loss.append(loss.item())
                        lr_scheduler.step()
                        current_lr = lr_scheduler.get_last_lr()[0]  # 获取最新的学习率
                        print('epoch:', epoch, loss)

                        if (epoch + 1) % args.eval_freq == 0:
                            print("Epoch {:04d} , average loss {:.4f} , epoch_time {:.4f}\n".format(
                                epoch + 1, sum(epoch_loss) / len(epoch_loss), time.time() - t))
                            model.eval()
                            with torch.no_grad():
                                val_metrics = corpus.get_validation_pred(model, 'test')
                            if val_metrics['Mean Reciprocal Rank'] > best_test_metrics['Mean Reciprocal Rank']:
                                best_test_metrics['Mean Reciprocal Rank'] = val_metrics['Mean Reciprocal Rank']
                            if val_metrics['Mean Rank'] < best_test_metrics['Mean Rank']:
                                best_test_metrics['Mean Rank'] = val_metrics['Mean Rank']
                            if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                                best_test_metrics['Hits@1'] = val_metrics['Hits@1']
                            if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                                best_test_metrics['Hits@3'] = val_metrics['Hits@3']
                            if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                                best_test_metrics['Hits@10'] = val_metrics['Hits@10']
                            if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                                best_test_metrics['Hits@100'] = val_metrics['Hits@100']
                            print(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                                            model.format_metrics(val_metrics, 'test')]))
                            print(f'Epoch {epoch + 1}/{args.epochs}, Current learning rate: {current_lr}')
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f'Error at epoch {epoch}: {e}')
                        print(f'Error type: {type(e).__name__}')
                        print(''.join(traceback.format_exception(None, e, e.__traceback__)))
                        break

                print('Optimization Finished!')
                print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
                if not best_test_metrics:
                    model.eval()
                    with torch.no_grad():
                        best_test_metrics = corpus.get_validation_pred(model, 'test')
                print(' '.join(['Val set results:',
                                model.format_metrics(best_val_metrics, 'val')]))
                print(' '.join(['Test set results:',
                                model.format_metrics(best_test_metrics, 'test')]))

                results[config_key] = best_test_metrics

                '''if args.save:
                    torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}.pth')
                    print('Saved model!')'''

    print("\nAll Configurations Results:")
    for config, metrics in results.items():
        print(config, ":", model.format_metrics(metrics, 'test'))



if __name__ == '__main__':
    #train_encoder(args)
    train_decoder(args)

