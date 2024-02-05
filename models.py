from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import utils
import random


class Unlabeled_Dataset(Dataset):
    def __init__(self, tokenized_users, domain_label=0):
        self._user_input_ids = []

        for item in tokenized_users:
            self._user_input_ids.append(tokenized_users[item])

        self._domain_labels = [domain_label] * len(self._user_input_ids)
        self._domain_labels = torch.tensor(self._domain_labels)

    def __len__(self):
        return len(self._user_input_ids)

    def __getitem__(self, idx):
        return self._user_input_ids[idx], self._domain_labels[idx]


class Ratings_Dataset(Dataset):
    def __init__(self, u_i_ratings, u_ids_source, u_ids_target, p_ids, isTrain=False, u_reviews_source=None,
                 u_reviews_target=None, p_reviews=None):
        self._user_input_ids_source = []
        self._user_input_ids_target = []
        self._product_input_ids = []
        self._labels = []
        self._u_rs_source = []
        self._u_rs_target = []
        self._p_rs = []

        for item in u_i_ratings:
            user = item[0]
            product = item[1]
            rating = item[2]

            # if p_reviews[product] == "This item has no reviews":
            #     # continue
            #     pass

            # if u_reviews_source:
            #     self._u_rs_source.append(u_reviews_source[user][0])
            #     # print(user in u_reviews_target)
            #     self._u_rs_target.append(u_reviews_target[user][0])
            #     self._p_rs.append(p_reviews[product])

            if not isTrain:
                # self._usernames.append(user)
                self._user_input_ids_source.append(u_ids_source[user][0])

                self._user_input_ids_target.append(u_ids_target[user][0])

                self._product_input_ids.append(p_ids[product])

                self._u_rs_source.append(u_reviews_source[user][0])
                # print(user in u_reviews_target)
                self._u_rs_target.append(u_reviews_target[user][0])
                self._p_rs.append(p_reviews[product])

                self._labels.append(rating)
            else:
                # need logic to check if there exits such many reviews when # of reviews is large
                for idx in range(len(u_ids_source[user])):
                    # self._usernames.append(user)
                    self._user_input_ids_source.append(u_ids_source[user][idx])
                    self._user_input_ids_target.append(u_ids_target[user][idx])

                    self._product_input_ids.append(p_ids[product])
                    self._labels.append(rating)

                    self._u_rs_source.append(u_reviews_source[user][0])
                    # print(user in u_reviews_target)
                    self._u_rs_target.append(u_reviews_target[user][0])
                    self._p_rs.append(p_reviews[product])

        self._labels = torch.tensor(self._labels)

        # print(len(self._u_rs_source))
        # print(len(self._user_input_ids_source))
        print('************')
        print(len(self._p_rs))
        print(len(self._u_rs_target))
        print(len(self._u_rs_source))
        print(len(self._user_input_ids_source))
        print('************')

    def __len__(self):
        return len(self._user_input_ids_source)

    def __getitem__(self, idx):
        return self._user_input_ids_source[idx], self._user_input_ids_target[idx], self._product_input_ids[idx], \
        self._labels[idx], self._u_rs_source[idx], self._u_rs_target[idx], self._p_rs[idx]

class SimpleBaseLine(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[200, 200, 200],
                 num_classes=1,
                 dropout=0.5):

        super(SimpleBaseLine, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_list_user_source = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.conv1d_list_user_target = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.conv1d_list_product = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.dropout = nn.Dropout(p=dropout)

        self._common_layer1_size = 256
        self._common_layer2_size = 128
        self._projection_size = 128
        self._domain_classifier_layer1_size = 64
        self._product_layer_size = 128
        self.relu = nn.ReLU()

        self.product_layer = nn.Linear(np.sum(num_filters), self._product_layer_size)
        self.rating_layer = nn.Linear(self._product_layer_size + self._common_layer2_size, num_classes)
        self.projection_layer = nn.Linear(self._product_layer_size + self._common_layer2_size, self._projection_size)

        self.common_features_layer = nn.Linear(np.sum(num_filters), self._common_layer2_size)
        # self.common_features_layer_fc1 = nn.Linear(np.sum(num_filters), self._common_layer1_size)
        # self.common_features_layer = nn.Linear(self._common_layer1_size, self._common_layer2_size)

        self.domain_classifier = nn.Linear(self._common_layer2_size, 2)
        # self.domain_classifier = nn.Linear(self._domain_classifier_layer1_size, 2)
        # self.domain_classifier_fc1 = nn.Linear(self._common_layer2_size, self._domain_classifier_layer1_size)

    def get_user_embeddings(self, user_input_ids_source, user_input_ids_target):
        user_embed_source = self.embedding(user_input_ids_source).float()
        user_embed_target = self.embedding(user_input_ids_target).float()

        user_reshaped_source = user_embed_source.permute(0, 2, 1)
        user_reshaped_target = user_embed_target.permute(0, 2, 1)

        user_conv_list_source = [F.relu(conv1d(user_reshaped_source)) for conv1d in self.conv1d_list_user_source]
        user_conv_list_target = [F.relu(conv1d(user_reshaped_target)) for conv1d in self.conv1d_list_user_target]

        user_pool_list_source = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                                 for x_conv in user_conv_list_source]
        user_pool_list_target = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                                 for x_conv in user_conv_list_target]

        user_fc_source = torch.cat([x_pool.squeeze(dim=2) for x_pool in user_pool_list_source],
                                   dim=1)
        user_fc_target = torch.cat([x_pool.squeeze(dim=2) for x_pool in user_pool_list_target],
                                   dim=1)

        return user_fc_source, user_fc_target

    def get_product_embeddings(self, product_input_ids):
        product_embed = self.embedding(product_input_ids).float()
        product_reshaped = product_embed.permute(0, 2, 1)
        product_conv_list = [F.relu(conv1d(product_reshaped)) for conv1d in self.conv1d_list_product]
        product_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                             for x_conv in product_conv_list]
        product_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in product_pool_list],
                               dim=1)

        return product_fc

    def forward(self, user_input_ids_source, user_input_ids_target, product_input_ids=None, alpha=0):

        if product_input_ids is not None:
            user_fc_source, user_fc_target = self.get_user_embeddings(user_input_ids_source, user_input_ids_target)
            product_fc = self.get_product_embeddings(product_input_ids)

            product_fc = self.product_layer(self.relu(self.dropout(product_fc)))

            # common_feature_source = self.common_features_layer_fc1(self.relu(self.dropout(user_fc_source)))
            common_feature_target = self.common_features_layer(self.relu(self.dropout(user_fc_target)))

            # common_feature_source = self.common_features_layer(self.relu(self.dropout(common_feature_source)))
            # common_feature_target = self.common_features_layer(self.relu(self.dropout(common_feature_target)))

            # source 是否需要取mask？ 还是整个source都用来计算loss
            # 所有训练数据都有 source 跟 target reviews，所以不需要用mask抽取不含target部分
            # source_concat = torch.cat((common_feature_source, product_fc), 1)
            target_concat = torch.cat((common_feature_target, product_fc), 1)

            rating_output = self.rating_layer(self.relu(self.dropout(target_concat)))

            # projection_source = self.projection_layer(self.relu(self.dropout(source_concat)))
            # projection_target = self.projection_layer(self.relu(self.dropout(target_concat)))

            projection_source = None
            projection_target = None

            return rating_output, projection_source, projection_target
        else:
            user_fc_source, user_fc_target = self.get_user_embeddings(user_input_ids_source, user_input_ids_target)

            # Extract common features from the user's source embedding and target embedding
            common_feature_source = self.common_features_layer(self.relu(self.dropout(user_fc_source)))
            common_feature_target = self.common_features_layer(self.relu(self.dropout(user_fc_target)))

            # common_feature_source = self.common_features_layer(self.relu(self.dropout(common_feature_source)))
            # common_feature_target = self.common_features_layer(self.relu(self.dropout(common_feature_target)))

            reverse_source_feature = utils.ReverseLayerF.apply(common_feature_source, alpha)
            reverse_target_feature = utils.ReverseLayerF.apply(common_feature_target, alpha)

            # source_domain_output = self.domain_classifier_fc1(self.relu(self.dropout(reverse_source_feature)))
            # target_domain_output = self.domain_classifier_fc1(self.relu(self.dropout(reverse_target_feature)))

            # source_domain_output = self.domain_classifier(self.relu(self.dropout(source_domain_output)))
            # target_domain_output = self.domain_classifier(self.relu(self.dropout(target_domain_output)))
            source_domain_output = self.domain_classifier(self.relu(self.dropout(reverse_source_feature)))
            target_domain_output = self.domain_classifier(self.relu(self.dropout(reverse_target_feature)))

            return source_domain_output, target_domain_output

class CNN_SCL_GRL_DS(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[200, 200, 200],
                 num_classes=1,
                 dropout=0.5):

        super(CNN_SCL_GRL_DS, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_list_user_source = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.conv1d_list_user_target = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.conv1d_list_product = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.dropout = nn.Dropout(p=dropout)

        # self._common_layer1_size = 256
        self._common_layer2_size = 64
        self._projection_size = 128
        self._domain_classifier_layer1_size = 64
        self._product_layer_size = 128
        self.relu = nn.ReLU()

        self.product_layer = nn.Linear(np.sum(num_filters), self._product_layer_size)
        self.rating_layer = nn.Linear(self._product_layer_size + self._common_layer2_size * 2, num_classes)
        self.projection_layer = nn.Linear(self._product_layer_size + self._common_layer2_size * 2, self._projection_size)

        self.common_features_layer = nn.Linear(np.sum(num_filters), self._common_layer2_size)
        self.source_specific_layer = nn.Linear(np.sum(num_filters), self._common_layer2_size)
        self.target_specific_layer = nn.Linear(np.sum(num_filters), self._common_layer2_size)

        self.domain_classifier = nn.Linear(self._common_layer2_size, 2)
        self.source_domain_classifier = nn.Linear(self._common_layer2_size, 2)
        self.target_domain_classifier = nn.Linear(self._common_layer2_size, 2)


    def get_user_embeddings(self, user_input_ids_source, user_input_ids_target):
        user_embed_source = self.embedding(user_input_ids_source).float()
        user_embed_target = self.embedding(user_input_ids_target).float()

        user_reshaped_source = user_embed_source.permute(0, 2, 1)
        user_reshaped_target = user_embed_target.permute(0, 2, 1)

        user_conv_list_source = [F.relu(conv1d(user_reshaped_source)) for conv1d in self.conv1d_list_user_source]
        user_conv_list_target = [F.relu(conv1d(user_reshaped_target)) for conv1d in self.conv1d_list_user_target]

        user_pool_list_source = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                                 for x_conv in user_conv_list_source]
        user_pool_list_target = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                                 for x_conv in user_conv_list_target]

        user_fc_source = torch.cat([x_pool.squeeze(dim=2) for x_pool in user_pool_list_source],
                                   dim=1)
        user_fc_target = torch.cat([x_pool.squeeze(dim=2) for x_pool in user_pool_list_target],
                                   dim=1)

        return user_fc_source, user_fc_target

    def get_product_embeddings(self, product_input_ids):
        product_embed = self.embedding(product_input_ids).float()
        product_reshaped = product_embed.permute(0, 2, 1)
        product_conv_list = [F.relu(conv1d(product_reshaped)) for conv1d in self.conv1d_list_product]
        product_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                             for x_conv in product_conv_list]
        product_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in product_pool_list],
                               dim=1)

        return product_fc

    def forward(self, user_input_ids_source, user_input_ids_target, product_input_ids=None, alpha=0):

        if product_input_ids is not None:
            user_fc_source, user_fc_target = self.get_user_embeddings(user_input_ids_source, user_input_ids_target)
            product_fc = self.get_product_embeddings(product_input_ids)

            product_fc = self.product_layer(self.relu(self.dropout(product_fc)))

            common_feature_source = self.common_features_layer(self.relu(self.dropout(user_fc_source)))
            common_feature_target = self.common_features_layer(self.relu(self.dropout(user_fc_target)))

            source_specific_features = self.source_specific_layer(self.relu(self.dropout(user_fc_source)))
            target_specific_features = self.target_specific_layer(self.relu(self.dropout(user_fc_target)))

            source_all_features = torch.cat((common_feature_source, source_specific_features), 1)
            target_all_features = torch.cat((common_feature_target, target_specific_features), 1)
            source_concat = torch.cat((source_all_features, product_fc), 1)
            target_concat = torch.cat((target_all_features, product_fc), 1)

            rating_output = self.rating_layer(self.relu(self.dropout(target_concat)))

            projection_source = self.projection_layer(self.relu(self.dropout(source_concat)))
            projection_target = self.projection_layer(self.relu(self.dropout(target_concat)))

            return rating_output, projection_source, projection_target
        else:
            user_fc_source, user_fc_target = self.get_user_embeddings(user_input_ids_source, user_input_ids_target)

            # Extract common features from the user's source embedding and target embedding
            common_feature_source = self.common_features_layer(self.relu(self.dropout(user_fc_source)))
            common_feature_target = self.common_features_layer(self.relu(self.dropout(user_fc_target)))

            reverse_source_feature = utils.ReverseLayerF.apply(common_feature_source, alpha)
            reverse_target_feature = utils.ReverseLayerF.apply(common_feature_target, alpha)

            source_domain_output = self.domain_classifier(self.relu(self.dropout(reverse_source_feature)))
            target_domain_output = self.domain_classifier(self.relu(self.dropout(reverse_target_feature)))

            # Extract domain specific features from the user's source embedding and target embedding
            source_specific_features = self.source_specific_layer(self.relu(self.dropout(user_fc_source)))
            target_specific_features = self.target_specific_layer(self.relu(self.dropout(user_fc_target)))

            source_specific_domain_output = self.source_domain_classifier(self.relu(self.dropout(source_specific_features)))
            target_specific_domain_output = self.target_domain_classifier(self.relu(self.dropout(target_specific_features)))

            return source_domain_output, target_domain_output, source_specific_domain_output, target_specific_domain_output

class Baseline_DeepCoNN(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[200, 200, 200],
                 num_classes=1,
                 dropout=0.5):

        super(Baseline_DeepCoNN, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_list_user_source = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.conv1d_list_product = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.dropout = nn.Dropout(p=dropout)

        self._common_layer2_size = 64
        self._projection_size = 128
        self._domain_classifier_layer1_size = 64
        self._product_layer_size = 128
        self.relu = nn.ReLU()

        self.product_layer = nn.Linear(np.sum(num_filters), self._product_layer_size)
        self.rating_layer = nn.Linear(self._product_layer_size + self._common_layer2_size * 2, num_classes)
        self.projection_layer = nn.Linear(self._product_layer_size + self._common_layer2_size * 2, self._projection_size)

        self.common_features_layer = nn.Linear(np.sum(num_filters), self._common_layer2_size)
        self.source_specific_layer = nn.Linear(np.sum(num_filters), self._common_layer2_size)
        self.target_specific_layer = nn.Linear(np.sum(num_filters), self._common_layer2_size)

        self.domain_classifier = nn.Linear(self._common_layer2_size, 2)
        self.source_domain_classifier = nn.Linear(self._common_layer2_size, 2)
        self.target_domain_classifier = nn.Linear(self._common_layer2_size, 2)

    def get_user_embeddings(self, user_input_ids_source):
        user_embed_source = self.embedding(user_input_ids_source).float()

        user_reshaped_source = user_embed_source.permute(0, 2, 1)

        user_conv_list_source = [F.relu(conv1d(user_reshaped_source)) for conv1d in self.conv1d_list_user_source]

        user_pool_list_source = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                                 for x_conv in user_conv_list_source]

        user_fc_source = torch.cat([x_pool.squeeze(dim=2) for x_pool in user_pool_list_source],
                                   dim=1)

        return user_fc_source

    def get_product_embeddings(self, product_input_ids):
        product_embed = self.embedding(product_input_ids).float()
        product_reshaped = product_embed.permute(0, 2, 1)
        product_conv_list = [F.relu(conv1d(product_reshaped)) for conv1d in self.conv1d_list_product]
        product_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                             for x_conv in product_conv_list]
        product_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in product_pool_list],
                               dim=1)

        return product_fc

    def forward(self, user_input_ids_source, product_input_ids=None):

        if product_input_ids is not None:
            user_fc_source = self.get_user_embeddings(user_input_ids_source)
            product_fc = self.get_product_embeddings(product_input_ids)

            product_fc = self.product_layer(self.relu(self.dropout(product_fc)))

            common_feature_source = self.common_features_layer(self.relu(self.dropout(user_fc_source)))

            source_specific_features = self.source_specific_layer(self.relu(self.dropout(user_fc_source)))

            source_all_features = torch.cat((common_feature_source, source_specific_features), 1)
            source_concat = torch.cat((source_all_features, product_fc), 1)

            rating_output = self.rating_layer(self.relu(self.dropout(source_concat)))

            return rating_output