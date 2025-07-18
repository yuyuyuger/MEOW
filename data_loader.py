import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Corpus:
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        self.device = args.device
        self.train_triples = train_data[0]
        self.val_triples = val_data[0]
        self.test_triples = test_data[0]
        self.max_batch_num = 1

        adj_indices = torch.LongTensor([train_data[1][0], train_data[1][1]])
        adj_values = torch.LongTensor([train_data[1][2]])
        self.train_adj_matrix = (adj_indices, adj_values)

        self.entity2id = {k: v for k, v in entity2id.items()}
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.relation2id = {k: v for k, v in relation2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.batch_size = args.batch_size

    def shuffle(self):
        raise NotImplementedError

    def get_batch(self, batch_num):
        raise NotImplementedError

    def get_validation_pred(self, model, split='test'):
        raise NotImplementedError


class ConvECorpus(Corpus):
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        super(ConvECorpus, self).__init__(args, train_data, val_data, test_data, entity2id, relation2id)
        rel_num = len(relation2id)
        for k, v in relation2id.items():
            self.relation2id[k+'_reverse'] = v+rel_num
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        sr2o = {}
        for (head, relation, tail) in self.train_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)

        self.triples = {}
        self.train_indices = [{'triple': (head, relation, -1), 'label': list(sr2o[(head, relation)])}
                              for (head, relation), tail in sr2o.items()]
        self.triples['train'] = [{'triple': (head, relation, -1), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation), tail in sr2o.items()]

        if len(self.train_indices) % self.batch_size == 0:
            self.max_batch_num = len(self.train_indices) // self.batch_size
        else:
            self.max_batch_num = len(self.train_indices) // self.batch_size + 1

        for (head, relation, tail) in self.val_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)

        for (head, relation, tail) in self.test_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)  #  sr2o 里是全部的（head，relation）：tail

        self.val_head_indices = [{'triple': (tail, relation + rel_num, head), 'label': list(sr2o[(tail, relation + rel_num)])}
                                 for (head, relation, tail) in self.val_triples]
        self.val_tail_indices = [{'triple': (head, relation, tail), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation, tail) in self.val_triples]
        self.test_head_indices = [{'triple': (tail, relation + rel_num, head), 'label': list(sr2o[(tail, relation + rel_num)])}
                                 for (head, relation, tail) in self.test_triples]
        self.test_tail_indices = [{'triple': (head, relation, tail), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation, tail) in self.test_triples]

    def read_batch(self, batch):
        triple, label = [_.to(self.device) for _ in batch]
        return triple, label

    def shuffle(self):
        np.random.shuffle(self.train_indices)

    def get_batch(self, batch_num):
        if (batch_num + 1) * self.batch_size <= len(self.train_indices):
            batch = self.train_indices[batch_num * self.batch_size: (batch_num+1) * self.batch_size]
        else:
            batch = self.train_indices[batch_num * self.batch_size:]
        batch_indices = torch.LongTensor([indice['triple'] for indice in batch])   # batch_indices：(head, relation, -1)
        label = [np.int32(indice['label']) for indice in batch]    # label：tail
        y = np.zeros((len(batch), len(self.entity2id)), dtype=np.float32)
        for idx in range(len(label)):
            for l in label[idx]:
                y[idx][l] = 1.0
        y = 0.9 * y + (1.0 / len(self.entity2id))   # 由于每个样本可能有多个尾实体，因此 label[idx] 返回的是一个包含多个尾实体索引的列表。然后在内部循环中，对于 label[idx] 中的每个实体索引 l，将 y 中对应位置的值设为 1.0，表示该实体是正确的尾实体。
        batch_values = torch.FloatTensor(y)

        '''index = []
        for idx in range(len(label)):
            pos = label[idx]
            np.random.shuffle(pos)
            neg = np.int32(list(range((len(self.entity2id)))))
            np.random.shuffle(neg)
            if len(pos) >= 10:
                index.append(np.concatenate((pos[:10], neg[:90])))
            else:
                index.append(np.concatenate((pos, neg[:100-len(pos)])))

        y = torch.FloatTensor(y)
        index_np_array = np.array(index)  # 将列表转换为 numpy 数组
        index = torch.LongTensor(index_np_array)
        batch_values = torch.gather(y, dim=1, index=index)'''

        return batch_indices, batch_values#, index

    def get_validation_pred(self, model, split='test'):
        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_10_head, hits_at_10_tail = 0, 0
        hits_at_3_head, hits_at_3_tail = 0, 0
        hits_at_1_head, hits_at_1_tail = 0, 0

        if split == 'val':
            head_indices = self.val_head_indices
            tail_indices = self.val_tail_indices
        else:
            head_indices = self.test_head_indices
            tail_indices = self.test_tail_indices

        if len(head_indices) % self.batch_size == 0:
            max_batch_num = len(head_indices) // self.batch_size
        else:
            max_batch_num = len(head_indices) // self.batch_size + 1
        for batch_num in range(max_batch_num):
            if (batch_num + 1) * self.batch_size <= len(head_indices):
                head_batch = head_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
                tail_batch = tail_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
            else:
                head_batch = head_indices[batch_num * self.batch_size:]
                tail_batch = tail_indices[batch_num * self.batch_size:]

            head_batch_indices = torch.LongTensor([indice['triple'] for indice in head_batch])
            head_batch_indices = head_batch_indices.to(self.device)
            pred = model.forward(head_batch_indices)
            #pred = (pred[0] + pred[1] + pred[2] + pred[3]) / 4.0
            #pred = pred[0]
            label = [np.int32(indice['label']) for indice in head_batch]
            y = np.zeros((len(head_batch), len(self.entity2id)), dtype=np.float32)
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device)
            target = head_batch_indices[:, 2]
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target]
            pred = torch.where(y.byte(), torch.zeros_like(pred), pred)
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                scores = np.delete(scores, tar)
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                sorted_indices = np.argsort(-scores, kind='stable')
                ranks_head.append(np.where(sorted_indices == rand)[0][0]+1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

            tail_batch_indices = torch.LongTensor([indice['triple'] for indice in tail_batch])
            tail_batch_indices = tail_batch_indices.to(self.device)
            pred = model.forward(tail_batch_indices)
            #pred = (pred[0] + pred[1] + pred[2] + pred[3]) / 4.0
            #pred = pred[0]
            label = [np.int32(indice['label']) for indice in tail_batch]
            y = np.zeros((len(tail_batch), len(self.entity2id)), dtype=np.float32)
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device)
            target = tail_batch_indices[:, 2]
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target]
            pred = torch.where(y.byte(), torch.zeros_like(pred), pred)
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                scores = np.delete(scores, tar)
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                sorted_indices = np.argsort(-scores, kind='stable')
                ranks_tail.append(np.where(sorted_indices == rand)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])

        for i in range(len(ranks_head)):
            if ranks_head[i] <= 100:
                hits_at_100_head += 1
            if ranks_head[i] <= 10:
                hits_at_10_head += 1
            if ranks_head[i] <= 3:
                hits_at_3_head += 1
            if ranks_head[i] == 1:
                hits_at_1_head += 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail += 1
            if ranks_tail[i] <= 10:
                hits_at_10_tail += 1
            if ranks_tail[i] <= 3:
                hits_at_3_tail += 1
            if ranks_tail[i] == 1:
                hits_at_1_tail += 1

        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)

        hits_100_head = hits_at_100_head / len(ranks_head)
        hits_10_head = hits_at_10_head / len(ranks_head)
        hits_3_head = hits_at_3_head / len(ranks_head)
        hits_1_head = hits_at_1_head / len(ranks_head)
        mean_rank_head = sum(ranks_head) / len(ranks_head)
        mean_reciprocal_rank_head = sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)

        hits_100_tail = hits_at_100_tail / len(ranks_tail)
        hits_10_tail = hits_at_10_tail / len(ranks_tail)
        hits_3_tail = hits_at_3_tail / len(ranks_tail)
        hits_1_tail = hits_at_1_tail / len(ranks_tail)
        mean_rank_tail = sum(ranks_tail) / len(ranks_tail)
        mean_reciprocal_rank_tail = sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)

        #hits_100 = max(hits_100_head,  hits_100_tail)
        #hits_10 = max(hits_10_head, hits_10_tail)
        #hits_3 = max(hits_3_head, hits_3_tail)
        #hits_1 = max(hits_1_head, hits_1_tail)
        #mean_rank = max(sum(ranks_head) / len(ranks_head), sum(ranks_tail) / len(ranks_tail))
        #mean_reciprocal_rank = max(sum(reciprocal_ranks_head) / len(reciprocal_ranks_head),sum(
            #reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        hits_100 = (hits_at_100_head / len(ranks_head) + hits_at_100_tail / len(ranks_tail)) / 2
        hits_10 = (hits_at_10_head / len(ranks_head) + hits_at_10_tail / len(ranks_tail)) / 2
        hits_3 = (hits_at_3_head / len(ranks_head) + hits_at_3_tail / len(ranks_tail)) / 2
        hits_1 = (hits_at_1_head / len(ranks_head) + hits_at_1_tail / len(ranks_tail)) / 2
        mean_rank = (sum(ranks_head) / len(ranks_head) + sum(ranks_tail) / len(ranks_tail)) / 2
        mean_reciprocal_rank = (sum(reciprocal_ranks_head) / len(reciprocal_ranks_head) + sum(
            reciprocal_ranks_tail) / len(reciprocal_ranks_tail)) / 2

        metrics = {"Hits@100_head": hits_100_head, "Hits@10_head": hits_10_head, "Hits@3_head": hits_3_head,
                   "Hits@1_head": hits_1_head,
                   "Mean Rank_head": mean_rank_head, "Mean Reciprocal Rank_head": mean_reciprocal_rank_head,
                   "Hits@100_tail": hits_100_tail, "Hits@10_tail": hits_10_tail, "Hits@3_tail": hits_3_tail,
                   "Hits@1_tail": hits_1_tail,
                   "Mean Rank_tail": mean_rank_tail, "Mean Reciprocal Rank_tail": mean_reciprocal_rank_tail,
                   "Hits@100": hits_100, "Hits@10": hits_10, "Hits@3": hits_3, "Hits@1": hits_1,
                   "Mean Rank": mean_rank, "Mean Reciprocal Rank": mean_reciprocal_rank}

        return metrics


class ConvKBCorpus(Corpus):
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        super(ConvKBCorpus, self).__init__(args, train_data, val_data, test_data, entity2id, relation2id)
        self.neg_num = args.neg_num
        if len(self.train_triples) % self.batch_size == 0:
            self.max_batch_num = len(self.train_triples) // self.batch_size
        else:
            self.max_batch_num = len(self.train_triples) // self.batch_size + 1

        self.train_indices = np.array(self.train_triples).astype(np.int32)
        self.train_values = np.array([[1]] * len(self.train_triples)).astype(np.float32)
        self.val_indices = np.array(self.val_triples).astype(np.int32)
        self.val_values = np.array([[1]] * len(self.val_triples)).astype(np.float32)
        self.test_indices = np.array(self.test_triples).astype(np.int32)
        self.test_values = np.array([[1]] * len(self.test_triples)).astype(np.float32)

        self.unique_entities = [entity2id[i] for i in train_data[2]]
        self.all_triples = {j: i for i, j in enumerate(self.train_triples + self.val_triples + self.test_triples)}

        self.batch_indices = np.empty((self.batch_size * (self.neg_num + 1), 3)).astype(np.int32)
        self.batch_values = np.empty((self.batch_size * (self.neg_num + 1), 1)).astype(np.float32)

    def shuffle(self):
        np.random.shuffle(self.train_indices)

    def get_batch(self, batch_num):
        if (batch_num + 1) * self.batch_size <= len(self.train_indices):
            self.batch_indices = np.empty((self.batch_size * (self.neg_num + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((self.batch_size * (self.neg_num + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * batch_num, self.batch_size * (batch_num + 1))
            last_index = self.batch_size

        else:
            last_batch_size = len(self.train_indices) - self.batch_size * batch_num
            self.batch_indices = np.empty((last_batch_size * (self.neg_num + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((last_batch_size * (self.neg_num + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * batch_num, len(self.train_indices))
            last_index = last_batch_size

        self.batch_indices[:last_index, :] = self.train_indices[indices, :]
        self.batch_values[:last_index, :] = self.train_values[indices, :]
        random_entities = np.random.randint(0, len(self.entity2id), last_index * self.neg_num)
        self.batch_indices[last_index: (last_index * (self.neg_num + 1)), :] = np.tile(
            self.batch_indices[:last_index, :], (self.neg_num, 1))
        self.batch_values[last_index: (last_index * (self.neg_num + 1)), :] = np.tile(
            self.batch_values[:last_index, :], (self.neg_num, 1))

        for i in range(last_index):
            for j in range(self.neg_num // 2):
                current_index = i * (self.neg_num // 2) + j

                while(random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                      self.batch_indices[last_index + current_index, 2]) in self.all_triples.keys():
                    random_entities[current_index] = np.random.randint(0, len(self.entity2id))

                self.batch_indices[last_index + current_index, 0] = random_entities[current_index]
                self.batch_values[last_index + current_index, :] = [-1]
            for j in range(self.neg_num // 2):
                current_index = last_index * (self.neg_num // 2) + i * (self.neg_num // 2) + j

                while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                       random_entities[current_index]) in self.all_triples.keys():
                    random_entities[current_index] = np.random.randint(0, len(self.entity2id))

                self.batch_indices[last_index + current_index, 2] = random_entities[current_index]
                self.batch_values[last_index + current_index, :] = [-1]

        return self.batch_indices, self.batch_values


    def get_validation_pred(self, model, split='test', batch_size=16):
        entity_list = torch.tensor(list(self.entity2id.values()), dtype=torch.long).to(self.device)
        if split == 'val':
            split_triples = torch.tensor(self.val_triples, dtype=torch.long)
        elif split == 'test':
            split_triples = torch.tensor(self.test_triples, dtype=torch.long)

        split_triples = split_triples.to(self.device)

        dataset = TensorDataset(split_triples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_10_head, hits_at_10_tail = 0, 0
        hits_at_3_head, hits_at_3_tail = 0, 0
        hits_at_1_head, hits_at_1_tail = 0, 0

        for batch in dataloader:
            triples = batch[0]
            batch_size = triples.size(0)

            # Create a batch where each triple is expanded with all possible entities
            all_entities = entity_list.unsqueeze(0).repeat(batch_size, 1)
            triple_expanded_head = triples.unsqueeze(1).repeat(1, len(entity_list), 1)
            triple_expanded_tail = triples.unsqueeze(1).repeat(1, len(entity_list), 1)

            triple_expanded_head[:, :, 0] = all_entities
            triple_expanded_tail[:, :, 2] = all_entities

            # Flatten the expanded arrays for processing in batches
            triple_expanded_head = triple_expanded_head.view(-1, 3)
            triple_expanded_tail = triple_expanded_tail.view(-1, 3)

            # Filter out existing triples
            #mask_head = [(h.item(), r.item(), t.item()) not in self.all_triples for h, r, t in triple_expanded_head]
            #mask_tail = [(h.item(), r.item(), t.item()) not in self.all_triples for h, r, t in triple_expanded_tail]

            #valid_head = triple_expanded_head[mask_head]
            #valid_tail = triple_expanded_tail[mask_tail]

            # Predict and reshape to original batch format for ranks calculation
            predictions_head = model.predict(triple_expanded_head).view(batch_size, -1)
            predictions_tail = model.predict(triple_expanded_tail).view(batch_size, -1)

            # Calculate ranks and reciprocal ranks
            _, indices_head = predictions_head.sort(dim=1, descending=False)
            _, indices_tail = predictions_tail.sort(dim=1, descending=False)

            ranks_head_batch = (indices_head == triples[:, 0].unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
            ranks_tail_batch = (indices_tail == triples[:, 2].unsqueeze(1)).nonzero(as_tuple=True)[1] + 1

            ranks_head.extend(ranks_head_batch.tolist())
            ranks_tail.extend(ranks_tail_batch.tolist())

            reciprocal_ranks_head.extend([1.0 / r for r in ranks_head_batch])
            reciprocal_ranks_tail.extend([1.0 / r for r in ranks_tail_batch])

        # Calculate hit rates and mean ranks
        for rank in ranks_head:
            if rank <= 100:
                hits_at_100_head += 1
            if rank <= 10:
                hits_at_10_head += 1
            if rank <= 3:
                hits_at_3_head += 1
            if rank == 1:
                hits_at_1_head += 1

        for rank in ranks_tail:
            if rank <= 100:
                hits_at_100_tail += 1
            if rank <= 10:
                hits_at_10_tail += 1
            if rank <= 3:
                hits_at_3_tail += 1
            if rank == 1:
                hits_at_1_tail += 1

        total_ranks_head = len(ranks_head)
        total_ranks_tail = len(ranks_tail)

        metrics = {
            "Hits@100_head": hits_at_100_head / total_ranks_head,
            "Hits@10_head": hits_at_10_head / total_ranks_head,
            "Hits@3_head": hits_at_3_head / total_ranks_head,
            "Hits@1_head": hits_at_1_head / total_ranks_head,
            "Mean Rank_head": sum(ranks_head) / total_ranks_head,
            "Mean Reciprocal Rank_head": sum(reciprocal_ranks_head) / total_ranks_head,
            "Hits@100_tail": hits_at_100_tail / total_ranks_tail,
            "Hits@10_tail": hits_at_10_tail / total_ranks_tail,
            "Hits@3_tail": hits_at_3_tail / total_ranks_tail,
            "Hits@1_tail": hits_at_1_tail / total_ranks_tail,
            "Mean Rank_tail": sum(ranks_tail) / total_ranks_tail,
            "Mean Reciprocal Rank_tail": sum(reciprocal_ranks_tail) / total_ranks_tail,
            "Hits@100": (hits_at_100_head / total_ranks_head + hits_at_100_tail / total_ranks_tail) / 2,
            "Hits@10": (hits_at_10_head / total_ranks_head + hits_at_10_tail / total_ranks_tail) / 2,
            "Hits@3": (hits_at_3_head / total_ranks_head + hits_at_3_tail / total_ranks_tail) / 2,
            "Hits@1": (hits_at_1_head / total_ranks_head + hits_at_1_tail / total_ranks_tail) / 2,
            "Mean Rank": (sum(ranks_head) / total_ranks_head + sum(ranks_tail) / total_ranks_tail) / 2,
            "Mean Reciprocal Rank": (sum(reciprocal_ranks_head) / total_ranks_head + sum(
                reciprocal_ranks_tail) / total_ranks_tail) / 2
        }

        return metrics