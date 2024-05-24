import numpy as np
import torch
import torch.nn as nn
import utils
import dataPreprocessing
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.25)
parser.add_argument('--domain, type=string, default='Books_Movies')
args = parser.parse_args()
utils.set_seed(args.seed)
print('*************')
print(args.seed)
print('*************')
device = utils.get_device()

train_dataloader, eval_dataloader, test_dataloader, unlabled_source_dataloader, unlabled_target_dataloader, word2idx = dataPreprocessing.get_dataloaders_all_reviews(
    domain=arg.domain, bs=arg.bs, num_of_augmentation=0, mode='summary')

loss_fn_ce = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
SCL_criterion = utils.SupConLoss()

embeddings = utils.load_pretrained_vectors(word2idx, "./crawl-300d-2M.vec")
embeddings = torch.tensor(embeddings)

model, optimizer = utils.initilize_model_DS(pretrained_embedding=embeddings,
                                            freeze_embedding=False,
                                            learning_rate=0.02,
                                            dropout=0.4)

model.cuda()
# use multi-gpu
# model = nn.DataParallel(model)

epochs = 15

best_loss = 100
final_test_loss = 0
final_mae_loss = 0

len_dataloader = len(train_dataloader)
unlabel_source_iter = iter(unlabled_source_dataloader)
unlabel_target_iter = iter(unlabled_target_dataloader)

print('start training')

for epoch_i in range(epochs):

    print('epoch --- ' + str(epoch_i))

    model.train()
    total_loss = 0
    total_mse_loss = 0
    total_domain_loss = 0
    total_supcon_loss = 0
    total_specific_loss = 0

    i = 0

    print(len(train_dataloader))
    for step, batch in enumerate(tqdm(train_dataloader)):
        p = float(i + epoch_i * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        user_input_ids = batch[0].to(device)
        user_input_ids_target = batch[1].to(device)
        product_input_ids = batch[2].to(device)
        labels = batch[3].to(device)

        optimizer.zero_grad()

        model_output_1, representation_1, representation_2 = model(user_input_ids, user_input_ids_target,
                                                                   product_input_ids, alpha=alpha)

        feature1 = F.normalize(representation_1, dim=1)
        feature2 = F.normalize(representation_2, dim=1)
        features = torch.cat([feature1.unsqueeze(1), feature2.unsqueeze(1)], dim=1)

        mse_loss = loss_fn(model_output_1, labels.view(-1, 1))
        scl_loss = SCL_criterion(features, labels)

        mse_loss = loss_fn(model_output_1, labels.view(-1, 1))

        # Compute Domain Loss
        unlabled_source_input, unlabeld_source_labels = utils.endless_get_next_batch(unlabled_source_dataloader,
                                                                                     unlabel_source_iter)
        unlabled_target_input, unlabeld_target_labels = utils.endless_get_next_batch(unlabled_target_dataloader,
                                                                                     unlabel_target_iter)
        unlabled_source_input = unlabled_source_input.to(device)
        unlabled_target_input = unlabled_target_input.to(device)
        unlabeld_source_labels = unlabeld_source_labels.to(device)
        unlabeld_target_labels = unlabeld_target_labels.to(device)

        source_domain_output, target_domain_output, source_specific_domain_output, target_specific_domain_output = model(unlabled_source_input, unlabled_target_input, product_input_ids=None, alpha=alpha)
        source_domain_loss = loss_fn_ce(source_domain_output, unlabeld_source_labels)
        target_domain_loss = loss_fn_ce(target_domain_output, unlabeld_target_labels)

        source_specific_domain_loss = loss_fn_ce(source_specific_domain_output, unlabeld_source_labels)
        target_specific_domain_loss = loss_fn_ce(target_specific_domain_output, unlabeld_target_labels)

        sum_domain_loss = source_domain_loss + target_domain_loss
        sum_specific_loss = source_specific_domain_loss + target_specific_domain_loss

        domain_ratio = arg.beta
        sup_con_ratio = arg.alpha
        # combine all losses
        combined_loss = mse_loss + domain_ratio * sum_domain_loss + domain_ratio * sum_specific_loss + sup_con_ratio * scl_loss

        total_loss += combined_loss.item()
        total_mse_loss += mse_loss.item()
        total_domain_loss += sum_domain_loss.item()
        total_specific_loss += sum_specific_loss.item()

        combined_loss.backward()

        optimizer.step()
    i += 1

    print("avg train total loss: {0}".format(total_loss / len(train_dataloader)))
    print("avg train mse loss: {0}".format(total_mse_loss / len(train_dataloader)))
    print("avg train specific loss: {0}".format(total_specific_loss / len(train_dataloader)))
    print("avg train domain loss: {0}".format(total_domain_loss / len(train_dataloader)))

    # ### Testing

    eval_loss, mae_loss = utils.evaluate(eval_dataloader, model)
    print('current eval loss:{0}'.format(eval_loss))
    if eval_loss < best_loss:
        best_loss = eval_loss
        test_loss, test_mae_loss = utils.evaluate(test_dataloader, model)
        final_test_loss = test_loss
        final_mae_loss = test_mae_loss
        print('current best test accuracy:{0}'.format(test_loss))
        print('current best mae accuracy:{0}'.format(final_mae_loss))

print('final test loss:{0}'.format(final_test_loss))
print('final best mae accuracy:{0}'.format(final_mae_loss))
