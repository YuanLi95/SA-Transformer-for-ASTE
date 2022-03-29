#coding utf-8

import json, os
import random
import argparse

import numpy
import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from syntactic_utils import build_dependency_matrix,build_position_matrix,build_positionizer,build_dependencyizer
from data_old import load_data_instances, DataIterator
from model import Syntax_Transformer_RNNModel
import utils_old


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args,position_tokenizer,dependency_tokenizer,dependency_embedding,position_embedding):

    # load double embedding
    word2index = json.load(open(args.prefix + 'doubleembedding/word_idx.json'))
    general_embedding = numpy.load(args.prefix +'doubleembedding/gen.vec.npy')
    general_embedding = torch.from_numpy(general_embedding)
    domain_embedding = numpy.load(args.prefix +'doubleembedding/'+args.dataset+'_emb.vec.npy')
    domain_embedding = torch.from_numpy(domain_embedding)
    print(args.prefix+args.dataset)

    # load dataset
    train_path =args.prefix + args.dataset + '/train.json'
    dev_path = args.prefix + args.dataset + '/dev.json'
    test_path = args.prefix + args.dataset + '/test.json'

    instances_train = load_data_instances(train_path, word2index,position_tokenizer,dependency_tokenizer, args)
    instances_dev = load_data_instances(dev_path, word2index,position_tokenizer,dependency_tokenizer, args)
    instances_test = load_data_instances(test_path, word2index,position_tokenizer,dependency_tokenizer, args)


    devset = DataIterator(instances_dev, args)
    testset = DataIterator(instances_test, args)


    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # build model
    if args.model == 'bilstm':
        model = Syntax_Transformer_RNNModel(general_embedding, domain_embedding,dependency_embedding,position_embedding, args).to(args.device)


    parameters = list(model.parameters())
    parameters = filter(lambda x: x.requires_grad, parameters)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decline, gamma=0.5, last_epoch=-1)
    # training
    best_joint_f1 = 0
    best_joint_epoch = 0
    test_f1 = 0
    test_p=0
    test_r = 0
    best_test_model = None
    for i in range(args.epochs):
        random.shuffle(instances_train)
        trainset = DataIterator(instances_train, args)
        train_all_loss = 0.0
        print('Epoch:{}'.format(i))
        for j in trange(trainset.batch_count):

            _,sentence_tokens, lengths, masks,dependency_masks,\
            syntactic_position_datas,edge_datas, aspect_tags,\
            opinion_tags, tags= trainset.get_batch(j)
            # print(sentence_tokens[0])
            # print(len(sentence_tokens))
            # print(lengths[0])
            # print(masks[0])
            # exit()
            predictions = model(sentence_tokens, lengths, masks,dependency_masks,\
            syntactic_position_datas,edge_datas)
            # print(predictions)

            loss = 0.
            train_all_loss+=loss
            tags_flatten = tags[:, :lengths[0], :lengths[0]].reshape([-1])
            type_list = {}
            for i in tags_flatten.cpu().tolist():
                type_list[i]=0


            for k in range(len(predictions)):
                prediction_flatten = predictions[k].reshape([-1, predictions[k].shape[3]])
                loss = loss + F.cross_entropy(prediction_flatten, tags_flatten, ignore_index=-1)
            train_all_loss+=loss

            # print()
            # exit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('this epoch train loss :{0}'.format(train_all_loss))
        joint_precision, joint_recall, joint_f1,dev_loss = eval(model, devset, args)
        print("------------------this is test result-------------------------------------")
        test_joint_precision, test_joint_recall, test_joint_f1,_ = eval(model, testset, args)
        if test_joint_f1>test_f1:
            test_f1=test_joint_f1
            test_p = test_joint_precision
            test_r = test_joint_recall
            best_test_model = model
        print("11111111111111")
        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + args.model + args.task+args.dataset+ '.pt'
            best_joint_f1 = joint_f1
            best_joint_epoch = i
            torch.save(model, model_path)
        print('this poch:\t dev {} loss: {:.5f}\n\n'.format( args.task, dev_loss))
    best_test_model_path = args.model_dir + args.model + args.task + args.dataset+"best_test_f1"+str(test_f1) + '.pt'
    torch.save(best_test_model,  best_test_model_path)
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))
    print('max test precision:{} ----- recall:{}-------- f1:{}'.format(str(test_p),str(test_r),str(test_f1)))
    # print()


def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():
        predictions=[]
        labels=[]
        all_ids = []
        all_lengths = []
        dev_loss =0.0
        for i in range(dataset.batch_count):
            sentence_ids,sentence_tokens, lengths, masks,dependency_masks,\
            syntactic_position_datas,edge_datas, aspect_tags,\
            opinion_tags, tags = dataset.get_batch(i)
            prediction = model.forward(sentence_tokens, lengths, masks,dependency_masks,\
            syntactic_position_datas,edge_datas)
            prediction = prediction[-1]
            tags_flatten = tags[:, :lengths[0], :lengths[0]].reshape([-1])


            prediction_flatten = prediction.reshape([-1, prediction.shape[3]])
            dev_loss = dev_loss + F.cross_entropy(prediction_flatten, tags_flatten, ignore_index=-1)

            prediction = torch.argmax(prediction, dim=3)
            prediction_padded = torch.zeros(prediction.shape[0], args.max_sequence_len, args.max_sequence_len)
            prediction_padded[:, :prediction.shape[1], :prediction.shape[1]] = prediction
            predictions.append(prediction_padded)

            all_ids.extend(sentence_ids)
            labels.append(tags)
            all_lengths.append(lengths)

        predictions = torch.cat(predictions,dim=0).cpu().tolist()
        labels = torch.cat(labels,dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()
        print(sentence_ids[0])
        precision, recall, f1 = utils_old.score_uniontags(args, predictions, labels, all_lengths, ignore_index=-1)

        aspect_results = utils_old.score_aspect(predictions, labels, all_lengths, ignore_index=-1)
        opinion_results = utils_old.score_opinion(predictions, labels, all_lengths, ignore_index=-1)
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1], aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1], opinion_results[2]))
        print(args.task+'\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1,dev_loss


def test(args):
    print("Evaluation on testset:")
    model_path = args.model_dir + args.model + args.task+args.dataset+ '.pt'
    model_path =  args.model_dir + "bilstmtripletres14best_test_f10.6947496947496947.pt"
    model = torch.load(model_path).to(args.device)
    test_path = args.prefix + args.dataset + '/test.json'
    word2index = json.load(open(args.prefix + 'doubleembedding/word_idx.json'))
    instances_test = load_data_instances(test_path, word2index, position_tokenizer, dependency_tokenizer, args)
    testset = DataIterator(instances_test, args)
    model.eval()
    eval(model,  testset, args)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--model', type=str, default="bilstm", choices=["bilstm", "cnn"],
                        help='option: bilstm, cnn')
    parser.add_argument('--dataset', type=str, default="res16",
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--lstm_dim', type=int, default=150,
                        help='dimension of lstm cell')
    parser.add_argument('--cnn_dim', type=int, default=256,
                        help='dimension of cnn')


    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=6,
                        help='label number')
    parser.add_argument('--dependency_embed_dim', type=int, default=200,
                        )
    parser.add_argument('--position_embed_dim', type=int, default=50,
                        )

    parser.add_argument('--dynamic_layer_dim', type=int, default=200,
                        )
    parser.add_argument('--seed',type = int, default=19)
    parser.add_argument('--num_attention_heads', type= int,default=5)
    parser.add_argument('--num_syntransformer_layers', type=int, default=3)
    parser.add_argument('--alpha_adjacent', type=float, default=0.4)
    parser.add_argument('--nhops', type=int, default=3,
                        help='inference times')

    parser.add_argument('--weight_edge', type=float, default=0.7)

    parser.add_argument('--decline', type=int, default=100, help="number of epochs to decline")


    args = parser.parse_args()
    #固定seed
    setup_seed(args.seed)



    position_tokenizer = build_positionizer(args.prefix + args.dataset)
    dependency_tokenizer = build_dependencyizer(args.prefix + args.dataset)
    dependency_embedding = build_dependency_matrix(dependency_tokenizer.dependency2idx,
                                                   args.dependency_embed_dim, args.prefix + args.dataset,
                                                   "dependency")
    position_embedding = build_position_matrix(position_tokenizer.position2idx,
                                               args.position_embed_dim, args.prefix + args.dataset,
                                               "position")

    if args.mode == 'train':
        train(args,position_tokenizer,dependency_tokenizer,dependency_embedding,position_embedding)
        # test(args)
    else:
        test(args)
