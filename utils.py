from torch.autograd import Function
import torch
import torch.nn as nn
import random
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch.optim as optim
import models
import string
import math


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # print(mask)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def endless_get_next_batch(loaders, iters):
    try:
        inputs, targets = next(iters)
    except StopIteration:
        iters = iter(loaders)
        inputs, targets = next(iters)

    return inputs, targets


def load_pretrained_w2v(word2idx, fname):
    """Load pretrained vectors and create embedding layers.

    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """
    print("Loading pretrained vectors...googleW2V")
    google_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    words = list(google_model.key_to_index.keys())
    word2vec = {word: google_model[word] % 300 for word in words}

    d = 300
    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    count = 0
    for word in words:
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = word2vec[word]
    # # Load pretrained vectors
    # count = 0
    # for line in tqdm(fin):
    #     tokens = line.rstrip().split(' ')
    #     word = tokens[0]
    #     if word in word2idx:
    #         count += 1
    #         embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings


def load_pretrained_vectors(word2idx, fname):
    """Load pretrained vectors and create embedding layers.

    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

    print("Loading pretrained vectors...")
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings


def encode(tokenized_texts, word2idx, max_len):
    """Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """

    input_ids = []
    for tokenized_sent in tokenized_texts:
        # Pad sentences to max_len
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        # Encode tokens to input_ids
        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_ids.append(input_id)

    return np.array(input_ids)


def tokenize(texts):
    """Tokenize texts, build vocabulary and find maximum sentence length.

    Args:
        texts (List[str]): List of text data

    Returns:
        tokenized_texts (List[List[str]]): List of list of tokens
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum sentence length
    """

    max_len = 0
    tokenized_texts = []
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    # Building our vocab from the corpus starting from index 2
    idx = 2
    for sent in texts:
        tokenized_sent = word_tokenize(sent)

        # Add `tokenized_sent` to `tokenized_texts`
        tokenized_texts.append(tokenized_sent)

        # Add new token to `word2idx`
        for token in tokenized_sent:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, word2idx, max_len


def remove_stopwords(sent):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    sent_list = sent.lower().split()
    count = 0
    while count < len(sent_list):
        if sent_list[count] in stop_words:
            sent_list.pop(count)
        else:
            count += 1

    return ' '.join(sent_list)


def set_seed(seed_value=1126):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def evaluate(evalation_dataloader, model):
    eval_total_loss = 0
    eval_mae_total_loss = 0

    device = get_device()
    loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(evalation_dataloader)):

            eval_user_input_ids = batch[0].to(device)
            eval_user_input_ids_target = batch[1].to(device)
            eval_product_input_ids = batch[2].to(device)
            eval_labels = batch[3].to(device)

            eval_model_output, _, _ = model(eval_user_input_ids, eval_user_input_ids_target, eval_product_input_ids)

            evaluation_loss = loss_fn(eval_model_output, eval_labels.view(-1,1))
            mae_loss = mae_loss_fn(eval_model_output, eval_labels.view(-1,1))

            eval_total_loss += evaluation_loss.item()
            eval_mae_total_loss += mae_loss.item()

    avg_eval_loss = eval_total_loss / len(evalation_dataloader)
    rmse_loss = math.sqrt(avg_eval_loss)

    avg_mae_loss = eval_mae_total_loss / len(evalation_dataloader)

    return rmse_loss, mae_loss


def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3,4,5],
                    num_filters=[200,200,200],
                    num_classes=1,
                    dropout=0.5,
                    learning_rate=0.01):

    cnn_model = models.CNN_SCL_GRL(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=dropout)

    # Send model to `device` (GPU/CPU)
    cnn_model.to(get_device())

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95)

    return cnn_model, optimizer


def initilize_model_DS(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3,4,5],
                    num_filters=[200,200,200],
                    num_classes=1,
                    dropout=0.5,
                    learning_rate=0.01):

    cnn_model = models.CNN_SCL_GRL_DS(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=dropout)

    # Send model to `device` (GPU/CPU)
    cnn_model.to(get_device())

    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95)

    return cnn_model, optimizer


def initilize_Baseline(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3,4,5],
                    num_filters=[200,200,200],
                    num_classes=1,
                    dropout=0.5,
                    learning_rate=0.01):

    cnn_model = models.Baseline_DeepCoNN(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=dropout)

    # Send model to `device` (GPU/CPU)
    cnn_model.to(get_device())

    # Instantiate Adadelta optimizer
    optimizer = optim.Adam(cnn_model.parameters(),
                               lr=learning_rate)

    return cnn_model, optimizer


def evaluate_DeepCoNN(evalation_dataloader, model):
    eval_total_loss = 0
    eval_mae_total_loss = 0

    device = get_device()
    loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(evalation_dataloader)):

            eval_user_input_ids = batch[0].to(device)
            eval_product_input_ids = batch[2].to(device)
            eval_labels = batch[3].to(device)

            eval_model_output = model(eval_user_input_ids, eval_product_input_ids)

            evaluation_loss = loss_fn(eval_model_output, eval_labels.view(-1,1))
            mae_loss = mae_loss_fn(eval_model_output, eval_labels.view(-1,1))

            eval_total_loss += evaluation_loss.item()
            eval_mae_total_loss += mae_loss.item()

    avg_eval_loss = eval_total_loss / len(evalation_dataloader)
    rmse_loss = math.sqrt(avg_eval_loss)

    avg_mae_loss = eval_mae_total_loss / len(evalation_dataloader)
    mae_loss = avg_mae_loss

    return rmse_loss, mae_loss


def evaluate_analysis(evalation_dataloader, model):
    eval_total_loss = 0

    device = get_device()
    loss_fn = nn.MSELoss()

    model.eval()

    total = 0
    count = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(evalation_dataloader)):

            # if step <= 10:

            eval_user_input_ids = batch[0].to(device)
            eval_user_input_ids_target = batch[1].to(device)
            eval_product_input_ids = batch[2].to(device)
            eval_labels = batch[3].to(device)
            u_rs_s = batch[4]
            u_rs_t = batch[5]
            p_rs = batch[6]

            eval_model_output, _, _ = model(eval_user_input_ids, eval_user_input_ids_target, eval_product_input_ids)

            evaluation_loss = loss_fn(eval_model_output, eval_labels.view(-1,1))

            eval_total_loss += evaluation_loss.item()


            eval_list = []
            for l in eval_model_output.view(-1, 1).tolist():
                eval_list.append(l[0])

            for i in range(len(eval_list)):
                # if p_rs[i] != 'This item has no reviews':
                total += (eval_list[i] - eval_labels[i]) ** 2
                count += 1

            # print('eval model output')
            # print(eval_list)
            # print('--------')
            # print('eval label output')
            # print(eval_labels.tolist())
            # print('--------')

            # count = 0
            # for i in range(len(eval_list)):
            #     if abs(eval_labels[i] - eval_list[i]) < 0.3 and eval_labels[i] == 5:
            #         # print(i)
            #         print('---')
            #         print(eval_list[i])
            #         print(eval_labels[i])
            #         print(u_rs_s[i])
            #         print('*******')
            #         print(u_rs_t[i])
            #         print('*******')
            #         print(p_rs[i])
            #         print('---')


    avg_eval_loss = eval_total_loss / len(evalation_dataloader)
    print('mse = ' + str(total/ count))

    return avg_eval_loss