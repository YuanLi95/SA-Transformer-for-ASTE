import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from attention_module import MultiHeadedAttention, SelfAttention
import  math
from  transformers.modeling_utils import ModuleUtilsMixin
class DynamicLayer(torch.nn.Module):



    def __init__(self,args):
        super(DynamicLayer, self).__init__()
        self.args = args
        self.linear_q = torch.nn.Linear(args.dependency_embed_dim,args.dynamic_layer_dim)
        self.linear_k = torch.nn.Linear(args.dependency_embed_dim, args.dynamic_layer_dim)
        self.linear_v = torch.nn.Linear(args.dependency_embed_dim, args.dynamic_layer_dim)
        self.alph_linear = torch.nn.Linear(2*args.dynamic_layer_dim, 1)

        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.dynamic_layer_dim/self.num_attention_heads )
        self.sig_1 = torch.nn.Linear(2*args.dynamic_layer_dim, 2*args.dynamic_layer_dim, bias=False)
        self.sig_2 = torch.nn.Linear(2*args.dynamic_layer_dim, 2*args.dynamic_layer_dim, bias=False)
    def transpose_for_edge(self, x):
        # print(x.shape)

        if len(x.shape) == 3:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)

            # return x.permute(0, 2, 1, 3)
            return x.permute(0, 3, 1, 2, 4)

    def forward(self,edge_embedding,dependency_masks):
        query = self.linear_q(edge_embedding)
        key  =self.linear_k(edge_embedding)
        value = self.linear_v(edge_embedding)
        # value =key
        query = self.transpose_for_edge(query)
        key = self.transpose_for_edge(key)
        value = self.transpose_for_edge(value)

        # print(query.shape)
        mask_row =dependency_masks
        # print()
        weights_row = torch.matmul(query, key.transpose(-1,-2))
        mask_muti = mask_row.unsqueeze(1)
        mask_muti = mask_muti.unsqueeze(-1)
        mask_muti  =torch.repeat_interleave(mask_muti,repeats=self.num_attention_heads,dim=1)
        # print(weights_row.shape)
        # print(mask_row.shape)

        weights_row = weights_row .masked_fill(mask_muti.expand_as(weights_row) == 0, float(-10000))
        attention_row = F.softmax(weights_row, dim=-1)/math.sqrt(self.attention_head_size)
        merged_row = torch.matmul(attention_row, value)

        merged_row = merged_row.permute(0, 2, 3, 1,4).contiguous()

        merged_row_shape = merged_row.size()[:-2] + (self.args.dynamic_layer_dim,)

        merged_row = merged_row.view(*merged_row_shape)
        merged_T = merged_row.transpose(1,2)


        alph = torch.sigmoid(self.alph_linear(torch.cat([merged_row,merged_T],dim=-1)))
        final_edge_feature = (1-alph)*merged_row+alph*merged_T
        final_edge_feature= final_edge_feature
        # print(final_edge_feature.shape)
        # print(mask_muti.shape)
        # final_edge_feature = final_edge_feature.masked_fill(dependency_masks.unsqueeze(-1).expand_as(final_edge_feature) == 0, 0)
        return  final_edge_feature

class Syntax_Transformer(torch.nn.Module):

    def __init__(self,args):
        super(Syntax_Transformer, self).__init__()
        self.hidden_size = int(args.lstm_dim*2)
        self.text_W_q = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.text_W_K = torch.nn.Linear(self.hidden_size,
                                        self.hidden_size)
        self.text_W_V = torch.nn.Linear(self.hidden_size,
                                   self.hidden_size)
        
        self.edge_k = torch.nn.Linear(args.dependency_embed_dim,
                                        self.hidden_size)
        self.edge_v = torch.nn.Linear(args.dependency_embed_dim,
                                      self.hidden_size)

        self.num_attention_heads= args. num_attention_heads
        self.attention_head_size = int((self.hidden_size)/self.num_attention_heads)
        # self.edge_k = torch.nn.Linear(args.dependency_embed_dim,
        #                               args.lstm_dim * 4 + args.position_embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.args = args

    def transpose_for_scores(self, x):
        if len(x.shape)==3:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        else:
            # print(x.size()[:-1])
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)

            # return x.permute(0, 2, 1, 3)
            return x.permute(0,3,1,2,4)

    def transpose_for_edge(self, x):
        # print(x.shape)

        if len(x.shape) == 3:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)

            # return x.permute(0, 2, 1, 3)
            return x.permute(0, 3, 1, 2, 4)

    def forward(self, token_feature,edge_feature,dependency_masks):
        batch, seq, dim = token_feature.shape
        mixed_query_layer = self.text_W_q(token_feature)
        mixed_key_layer = self.text_W_K(token_feature)
        mixed_value_layer =self.text_W_V(token_feature)
        # mixed_value_layer =mixed_key_layer


        mixed_query_layer = mixed_query_layer.unsqueeze(2)

        mixed_key_layer = mixed_key_layer.unsqueeze(1)
        mixed_key_layer = torch.repeat_interleave(mixed_key_layer, repeats=seq, dim=1)



        mixed_value_layer = mixed_value_layer.unsqueeze(1)
        mixed_value_layer = torch.repeat_interleave(mixed_value_layer, repeats=seq, dim=1)
        edge_mask = dependency_masks.unsqueeze(1)
        edge_mask = torch.repeat_interleave(edge_mask , repeats=self.num_attention_heads, dim=1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # print(value_layer.shape)

        edge_k = self.edge_k(edge_feature)
        edge_v = self.edge_v(edge_feature)
        edge_k = self.transpose_for_edge(edge_k)
        edge_v = self.transpose_for_edge(edge_v)
        # print(edge.shape)

        # print(query_layer.shape)
        # print(key_layer .shape)
        # print(value_layer.shape)
        # print(edge_v.shape)
        # print(edge_mask.shape)

        #hadamard product
        key_layer = key_layer+self.args.weight_edge*edge_k
        value_layer=value_layer+self.args.weight_edge*edge_v
        # key_layer = key_layer.mul(edge)
        # value_layer = value_layer.mul(edge)
        # print(key_layer.shape)
        # print(value_layer.shape)
        # print(query_layer.shape)


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)).squeeze(-2)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_weights = attention_scores.masked_fill(edge_mask.expand_as(attention_scores) == 0, float(-10000))


        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # print(attention_weights.shape)
        attention_probs = nn.Softmax(dim=-1)(attention_weights).unsqueeze(-2)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).squeeze(-2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer
        # outputs = token_feature+outputs
        outputs = self.LayerNorm(token_feature+outputs)
        return outputs



# edge_embedding batch *seq_len*seq_len*embedding_dim
# dependency_mask batch *seq_len*seq_len

class Syntax_Transformer_RNNModel(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb,dependency_emb, pos_emb, args):
        '''double embedding + lstm encoder + dot self attention'''
        super(Syntax_Transformer_RNNModel, self).__init__()

        self.args = args
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight.data.copy_(gen_emb)
        self.gen_embedding.weight.requires_grad = False

        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight.data.copy_(domain_emb)
        self.domain_embedding.weight.requires_grad = False

        self.dependency_embedding = \
        torch.nn.Embedding(dependency_emb.shape[0],dependency_emb.shape[1],padding_idx=0,)
        # self.dependency_embedding.weight.data.copy_(torch.rand([dependency_emb.shape[0], dependency_emb.shape[1]]))
        self.dependency_embedding.weight.data.copy_(torch.from_numpy(dependency_emb))
        self.dependency_embedding.weight.requires_grad = True

        self.position_embedding = \
            torch.nn.Embedding(pos_emb.shape[0], pos_emb.shape[1], padding_idx=0)
        self.position_embedding.weight.data.copy_(torch.from_numpy(pos_emb))
        self.position_embedding.weight.data.copy_(torch.rand([pos_emb.shape[0], pos_emb.shape[1]]))
        self.position_embedding.weight.requires_grad = True

        self.dynamic_layer = DynamicLayer(args)

        self.syntax_transformer =nn.ModuleList([Syntax_Transformer(args) for _ in range (args.num_syntransformer_layers)])

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout_ops = torch.nn.Dropout(0.5)
        self.dropout_edge =torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.0)

        self.bilstm = torch.nn.LSTM(400, args.lstm_dim,
                                    num_layers=1, batch_first=True, bidirectional=True)
        # print(args.lstm_dim)
        self.attention_layer = SelfAttention(args)
        self.feature_linear = torch.nn.Linear(args.lstm_dim*4+args.position_embed_dim + args.class_num*4, args.lstm_dim*4+args.position_embed_dim)
        self.cls_linear = torch.nn.Linear(args.lstm_dim*4+args.position_embed_dim, args.class_num)
        self.alpha_adjacent = args.alpha_adjacent
        self.trans_linear =torch.nn.Linear(args.class_num*3,args.class_num)



    def _get_embedding(self, sentence_tokens, mask):
        gen_embed = self.gen_embedding(sentence_tokens)
        domain_embed = self.domain_embedding(sentence_tokens)

        embedding = torch.cat([gen_embed, domain_embed], dim=2)
        embedding = self.dropout1(embedding)
        # print(embedding.shape)
        # print(mask.unsqueeze(2).float().expand_as(embedding).shape)
        # print( mask.unsqueeze(2).float())
        # print(mask.shape)
        # print( mask.unsqueeze(2).float().shape)
        # print("11111111111111111111")
        # exit()
        embedding = embedding * mask.unsqueeze(2).float().expand_as(embedding)
        # print((embedding * mask.unsqueeze(2)).shape)
        # print(embedding.shape)
        # exit()
        return embedding


    def other_embedding(self,syntactic_position_datas,edge_datas,batch_max_lengths,mask):
        syntactic_position_datas = syntactic_position_datas[:,0:batch_max_lengths,0:batch_max_lengths]
        edge_datas = edge_datas[:,0:batch_max_lengths,0:batch_max_lengths]
        position_embedding = self.position_embedding(syntactic_position_datas)
        edge_embedding = self.dependency_embedding(edge_datas)
        mask_new = mask[:,0:batch_max_lengths].unsqueeze(2).float()
        mask_T = mask_new.transpose(1, 2)

        mask_matrix = (mask_new*mask_T).unsqueeze(3).float()

        position_embedding = position_embedding*(mask_matrix.expand_as(position_embedding))
        edge_embedding = edge_embedding *(mask_matrix.expand_as(edge_embedding))

        edge_embedding = self.dropout_edge(edge_embedding)
        return position_embedding,edge_embedding, mask_matrix


    def _lstm_feature(self, embedding, lengths):
        # print(embedding)
        # print(lengths= )
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def _cls_logits(self, features):
        features = self.dropout2(features)
        # exit()
        tags = self.cls_linear(features)
        return tags

    def multi_hops(self, features, lengths, mask, k):
        '''generate mask'''
        max_length = features.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = mask.unsqueeze(3).expand([-1, -1, -1, self.args.class_num])
        # mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])
        logits = self._cls_logits(features)
        logits = logits
        left_logits = torch.zeros(logits.shape).to(self.args.device)
        up_logits = torch.zeros(logits.shape).to(self.args.device)
        dia_logits = torch.zeros(logits.shape).to(self.args.device)
        # print("1222222222222222222222")
        for i in range(k):
            probs = logits
            left_logits[:, 1:, :, :] = logits[:, 0:-1, :, :]
            up_logits[:, :, 1:, :] = logits[:, :, 0:-1, :]
            dia_logits[:, 1:, 1:, :] = logits[:, 0:-1, 0:-1, :]

            probs_T = probs.transpose(1,2)
            logits = probs * mask

            logits_a = torch.max(logits, dim=1)[0]
            logits_b = torch.max(logits, dim=2)[0]
            logits = torch.cat([logits_a.unsqueeze(3), logits_b.unsqueeze(3)], dim=3)
            logits = torch.max(logits, dim=3)[0]

            logits = logits.unsqueeze(2).expand([-1, -1, max_length, -1])
            logits_T = logits.transpose(1, 2)
            logits = torch.cat([logits, logits_T], dim=3)

            new_features = torch.cat([features, logits, probs,probs_T], dim=3)
            features = self.feature_linear(new_features)
            logits = self.cls_linear(features)

            # print(left_logits.shape)
            # print(up_logits)
            # print(dia_logits)
            other_logits = self.trans_linear (torch.cat([left_logits,up_logits,dia_logits],dim=-1))
            # other_logits = other_logits*mask
            logits_output= (1-self.alpha_adjacent)*logits+ self.alpha_adjacent*other_logits
            logits = logits_output
        return  logits
    # def Syntax_Transformer:



    def forward(self, sentence_tokens, lengths, mask,dependency_masks,syntactic_position_datas,edge_datas):
        # print(sentence_tokens.shape)
        # print(mask.shape)
        # print(lengths)
        # exit()
        batch_max_lengths = torch.max(lengths)

        token_embedding = self._get_embedding(sentence_tokens, mask)
        # print(token_embedding.shape)
        batch_size= token_embedding.shape[0]
        position_embedding,edge_embedding,mask_matrix = self.other_embedding(syntactic_position_datas,edge_datas,batch_max_lengths,mask)
        dependency_masks = dependency_masks[:,0:batch_max_lengths,0:batch_max_lengths]
        lstm_feature = self._lstm_feature(token_embedding, lengths)
        # self attention
        lstm_feature_attention = self.attention_layer(lstm_feature, lstm_feature, mask[:,:lengths[0]])

        #lstm_feature_attention = self.attention_layer.forward_perceptron(lstm_feature, lstm_feature, mask[:, :lengths[0]])
        lstm_feature = lstm_feature + lstm_feature_attention

        syn_feature = lstm_feature

        final_edge_feature = self.dynamic_layer(edge_embedding, dependency_masks)

        for layers in self.syntax_transformer:
            multi_feature = layers(syn_feature, final_edge_feature, dependency_masks)
            syn_feature = multi_feature
        lstm_feature = syn_feature
        # print(lengths)
        # exit()
        lstm_feature = lstm_feature.unsqueeze(2).expand([-1,-1, lengths[0], -1])

        lstm_feature_T = lstm_feature.transpose(1, 2)
        features = torch.cat([lstm_feature, lstm_feature_T], dim=3)




        concat_features = torch.cat([features,position_embedding],dim=3)


        # print(edge_embedding.shape)
        # print(mask_matrix.shape)
        # print(dependency_masks.shape)



        #
        #
        #
        # exit()

        logits = self.multi_hops(concat_features, lengths, mask, self.args.nhops)
        # print(logits[-1].shape)
        # exit()
        return [logits]
