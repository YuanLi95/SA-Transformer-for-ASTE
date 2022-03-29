import math

import torch
import json
import  os
import  pickle
sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}


def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class Instance(object):
    #sentence_pack,dependency_mask_data[i], edge_data[i],
     #                             syntax_position_data[i],word2index,position_tokenizer,
      #                            dependency_tokenizer,args)

    def __init__(self, sentence_pack,dependency_mask_pack,edge_data_pack,
                 syntax_position_pack,word2index,position_tokenizer,dependency_tokenizer, args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.sentence_tokens = torch.zeros(args.max_sequence_len).long()

        self.dependency_mask_seq = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.edge_data_seq = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.syntax_position_seq = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        position_matrix = position_tokenizer.position_to_index(syntax_position_pack)
        dependency_edge = dependency_tokenizer.dependency_to_index(edge_data_pack, dependency_mask_pack)

        '''generate sentence tokens'''
        words = self.sentence.split()
        self.length = len(words)
        self.dependency_mask_seq[0:self.length,0:self.length] = torch.from_numpy(dependency_mask_pack)
        self.syntax_position_seq[0:self.length,0:self.length] = torch.from_numpy(position_matrix)
        self.edge_data_seq[0:self.length,0:self.length] =torch.from_numpy(dependency_edge)



        for i, w in enumerate(words):
            # word = w.lower()
            word = w
            if word in word2index:
                self.sentence_tokens[i] = word2index[word]
            else:
                self.sentence_tokens[i] = word2index['<unk>']

        self.aspect_tags = torch.zeros(args.max_sequence_len).long()
        self.opinion_tags = torch.zeros(args.max_sequence_len).long()
        self.aspect_tags[self.length:] = -1
        self.opinion_tags[self.length:] = -1
        self.tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.tags[:, :] = -1

        for i in range(self.length):
            for j in range(i, self.length):
                self.tags[i][j] = 0
        for pair in sentence_pack['triples']:
            # print(pair)

            aspect = pair['target_tags']
            opinion = pair['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)
            # print(aspect_span)
            # print(opinion_span)
            # exit()
            for l, r in aspect_span:
                for i in range(l, r+1):
                    self.aspect_tags[i] = 1 if i == l else 2
                    self.tags[i][i] = 1
                    if i > l: self.tags[i-1][i] = 1
                    for j in range(i, r+1):
                        self.tags[i][j] = 1
            for l, r in opinion_span:
                for i in range(l, r+1):
                    self.opinion_tags[i] = 1 if i == l else 2
                    self.tags[i][i] = 2
                    if i > l: self.tags[i-1][i] = 2
                    for j in range(i, r+1):
                        self.tags[i][j] = 2
            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            if args.task == 'pair':
                                if i > j: self.tags[j][i] = 3
                                else: self.tags[i][j] = 3
                            elif args.task == 'triplet':
                                if i > j: self.tags[j][i] = sentiment2id[pair['sentiment']]
                                else: self.tags[i][j] = sentiment2id[pair['sentiment']]


            # print()

        '''generate mask of the sentence'''
        self.mask = torch.zeros(args.max_sequence_len)
        self.mask[:self.length] = 1


def load_data_instances(path, word2index,position_tokenizer,dependency_tokenizer, args):
    instances = list()
    sentence_packs = json.load(open(path))
    fout_undir_file = os.path.join("%sundir.graph" %path)
    syntax_position_file = os.path.join("%s.syntaxPosition" % path)
    dependency_type_file = os.path.join("%s.dependency" % path)
    dependency_undir = open(fout_undir_file, 'rb')
    edge_type = open(dependency_type_file, "rb")
    syntax_position = open(syntax_position_file, "rb")

    dependency_mask_data = pickle.load(dependency_undir)
    edge_data = pickle.load(edge_type)
    syntax_position_data = pickle.load(syntax_position)

    #
    dependency_undir.close()
    edge_type.close()
    syntax_position.close()

    for i,sentence_pack in enumerate(sentence_packs):
        instances.append(Instance(sentence_pack,dependency_mask_data[i], edge_data[i],
                                  syntax_position_data[i],word2index,position_tokenizer,
                                  dependency_tokenizer,args))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentence_tokens = []
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []
        dependency_masks = []
        syntactic_position_datas = []
        edge_datas = []
        # self.dependency_mask_seq = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        # self.edge_data_seq = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        # self.syntax_position_pack = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            sentence_tokens.append(self.instances[i].sentence_tokens)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)
            dependency_masks.append(self.instances[i].dependency_mask_seq)
            syntactic_position_datas.append(self.instances[i].syntax_position_seq)
            edge_datas.append(self.instances[i].edge_data_seq)

        indexes = list(range(len(sentence_tokens)))
        indexes = sorted(indexes, key=lambda x: lengths[x], reverse=True)

        sentence_ids = [sentence_ids[i] for i in indexes]
        sentence_tokens = torch.stack(sentence_tokens).to(self.args.device)[indexes]
        lengths = torch.tensor(lengths).to(self.args.device)[indexes]
        masks = torch.stack(masks).to(self.args.device)[indexes]
        aspect_tags = torch.stack(aspect_tags).to(self.args.device)[indexes]
        opinion_tags = torch.stack(opinion_tags).to(self.args.device)[indexes]
        tags = torch.stack(tags).to(self.args.device)[indexes]
        dependency_masks = torch.stack(dependency_masks).to(self.args.device)[indexes]
        syntactic_position_datas = torch.stack(syntactic_position_datas).to(self.args.device)[indexes]
        edge_datas = torch.stack(edge_datas).to(self.args.device)[indexes]

        return sentence_ids, sentence_tokens, lengths, masks,dependency_masks,\
               syntactic_position_datas,edge_datas, aspect_tags, opinion_tags, tags
