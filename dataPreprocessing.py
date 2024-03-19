import json
import random
import utils
import models
import aux_proc
from torch.utils.data import DataLoader

def train_eval_test_split(domains, split_ratio=1):
    if domains =="Movies_Music":
        domain1_file = './data/MoviesMusic_Movies_raw.json'
        domain2_file = './data/MoviesMusic_Music_raw.json'
    elif domains == "Music_Movies":
        domain1_file = './data/MusicMovies_Music_raw.json'
        domain2_file = './data/MusicMovies_Movies_raw.json'
    elif domains == "Books_Movies":
        domain1_file = './data/BooksMovies_Books_raw.json'
        domain2_file = './data/BooksMovies_Movies_raw.json'
    elif domains == "Movies_Books":
        domain1_file = './data/MoviesBooks_Movies_raw.json'
        domain2_file = './data/MoviesBooks_Books_raw.json'
    elif domains == "Books_Music":
        domain1_file = './data/BooksMusic_Books_raw.json'
        domain2_file = './data/BooksMusic_Music_raw.json'
    elif domains == "Music_Books":
        domain1_file = './data/MusicBooks_Music_raw.json'
        domain2_file = './data/MusicBooks_Books_raw.json'

    with open(domain1_file) as json_file:
        domain_1 = json.load(json_file)

    with open(domain2_file) as json_file:
        domain_2 = json.load(json_file)

    users = []
    for u in domain_1:
        users.append(u)

    random.shuffle(users)

    train_len = int(0.8 * len(users))
    eval_len = int(0.1 * len(users))

    fraction_training = int(train_len * split_ratio)
    train_users = users[:fraction_training]
    eval_users = users[train_len:train_len + eval_len]
    test_users = users[train_len + eval_len:]
    cold_start_users = eval_users + test_users

    print('# of train, eval, test users : {0}, {1}, {2}'.format(len(train_users), len(eval_users), len(test_users)))

    training_records_source = []
    training_records_target = []

    for u in train_users:
        for record in domain_1[u]:
            training_records_source.append(record)

    for u in eval_users:
        for record in domain_1[u]:
            training_records_source.append(record)

    for u in test_users:
        for record in domain_1[u]:
            training_records_source.append(record)

    #############
    for u in train_users:
        for record in domain_2[u]:
            training_records_target.append(record)

    eval_records = []
    for u in eval_users:
        for record in domain_2[u]:
            eval_records.append(record)

    test_records = []
    for u in test_users:
        for record in domain_2[u]:
            test_records.append(record)

    print('# of train(source), train(target), eval, test records : {0}, {1}, {2}, {3}'.format(len(training_records_source), len(training_records_target), len(eval_records), len(test_records)))

    return training_records_source, training_records_target, eval_records, test_records, cold_start_users


def get_unlabeled_data(domains, mode='summary'):
    if domains =="Movies_Music":
        domain1_file = './data/Movies_unlabeled.json'
        domain2_file = './data/Music_unlabeled.json'
    elif domains == "Music_Movies":
        domain1_file = './data/Music_unlabeled.json'
        domain2_file = './data/Movies_unlabeled.json'
    elif domains == "Books_Movies":
        domain1_file = './data/Books_unlabeled.json'
        domain2_file = './data/Movies_unlabeled.json'
    elif domains == "Movies_Books":
        domain1_file = './data/Movies_unlabeled.json'
        domain2_file = './data/Books_unlabeled.json'
    elif domains == "Books_Music":
        domain1_file = './data/Books_unlabeled.json'
        domain2_file = './data/Music_unlabeled.json'
    elif domains == "Music_Books":
        domain1_file = './data/Music_unlabeled.json'
        domain2_file = './data/Books_unlabeled.json'

    with open(domain1_file) as json_file:
        domain_1 = json.load(json_file)

    with open(domain2_file) as json_file:
        domain_2 = json.load(json_file)


    source_users = {}
    target_users = {}

    for record in domain_1:
        if 'reviewerID' not in record or mode not in record:
            continue

        if record['reviewerID'] not in source_users:
            source_users[record['reviewerID']] = record[mode]
        else:
            if len(source_users[record['reviewerID']].split()) < 200:
                source_users[record['reviewerID']] = source_users[record['reviewerID']] + ' ' + record[mode]

    for record in domain_2:
        if 'reviewerID' not in record or mode not in record:
            continue

        if record['reviewerID'] not in target_users:
            target_users[record['reviewerID']] = record[mode]
        else:
            if len(target_users[record['reviewerID']].split()) < 200:
                target_users[record['reviewerID']] = target_users[record['reviewerID']] + ' ' + record[mode]

    return source_users, target_users


def get_dataloaders_all_reviews(domain='Books_Movies', bs=64, num_of_augmentation=0, max_reviews_len=200, mode='summary'):
    train_source_list, train_target_list, eval_list, test_list, cold_start_users= train_eval_test_split(domain)

    unlabeled_source, unlabeled_target = get_unlabeled_data(domain, mode=mode)

    reviewer = set()
    for item in train_source_list + train_target_list + eval_list + test_list:
        reviewer.add(item['reviewerID'])

    print('number of all overlapping users: ' + str(len(reviewer)))

    product = set()
    for item in train_source_list + train_target_list + eval_list + test_list:
        product.add(item['asin'])

    print('number of all products: ' + str(len(product)))

    train_user_item_ratings_source = []
    for item in train_source_list:
        train_user_item_ratings_source.append([item['reviewerID'], item['asin'], item['overall']])

    train_user_item_ratings_target = []
    for item in train_target_list:
        train_user_item_ratings_target.append([item['reviewerID'], item['asin'], item['overall']])

    eval_user_item_ratings = []
    for item in eval_list:
        eval_user_item_ratings.append([item['reviewerID'], item['asin'], item['overall']])

    test_user_item_ratings = []
    for item in test_list:
        test_user_item_ratings.append([item['reviewerID'], item['asin'], item['overall']])

    user_reviews_source = {}
    user_reviews_target = {}
    product_reviews = {}

    print('start generating all users\' source representation')
    # generate all users' source representation
    for item in train_source_list + eval_list + test_list:
        if 'reviewerID' not in item or mode not in item:
            continue

        if item['reviewerID'] not in user_reviews_source:
            user_reviews_source[item['reviewerID']] = [utils.remove_stopwords(item[mode])] + [''] * num_of_augmentation
        else:

            for idx in range(num_of_augmentation + 1):
                if len(user_reviews_source[item['reviewerID']][idx].split()) < max_reviews_len:
                    user_reviews_source[item['reviewerID']][idx] = user_reviews_source[item['reviewerID']][idx] + ' ' + \
                                                                 utils.remove_stopwords(item[mode])
                    break

    print('start generating training users\' target representation')
    # generate training users'(eval/test users not included) target target representation
    for item in train_target_list:
        if 'reviewerID' not in item or mode not in item:
            continue

        if item['reviewerID'] not in user_reviews_target:
            user_reviews_target[item['reviewerID']] = [utils.remove_stopwords(item[mode])] + [''] * num_of_augmentation
        else:
            for idx in range(num_of_augmentation + 1):
                if len(user_reviews_target[item['reviewerID']][idx].split()) < max_reviews_len:
                    user_reviews_target[item['reviewerID']][idx] = user_reviews_target[item['reviewerID']][idx] + ' ' + \
                                                                 utils.remove_stopwords(item[mode])
                    break

    print('start generating eval/test users\' target representation')
    # generate eval/test users target representation
    # in summary mode, reviews are normally within 200 words
    cold_start_target_dict = aux_proc.generate_target_aux_doc(cold_start_users, domain=domain, mode=mode)

    for user in cold_start_target_dict:
        review = utils.remove_stopwords(cold_start_target_dict[user])
        review_splited = review.split()
        num_review = len(review_splited) // max_reviews_len + 1

        reviews_list = []
        reviews_list.append(' '.join(review_splited[0:max_reviews_len]))

        for idx in range(num_of_augmentation):
            if num_review >= idx + 2:
                reviews_list.append(' '.join(review_splited[max_reviews_len * (idx + 1): max_reviews_len * (idx + 2)]))
            else:
                reviews_list.append(reviews_list[-1])

        user_reviews_target[user] = reviews_list

    # generate products' representation
    for item in train_source_list + train_target_list:
        if item['asin'] not in product_reviews:
            product_reviews[item['asin']] = utils.remove_stopwords(item[mode])
        else:
            if len(product_reviews[item['asin']].split()) < max_reviews_len:
                product_reviews[item['asin']] = product_reviews[item['asin']] + ' ' + utils.remove_stopwords(item[mode])

    ##############################################################
    distinct_item_set = set()

    # for items with no reviews
    no_reviews_item_count = 0
    for item in eval_list + test_list:
        if 'reviewerID' not in item or mode not in item:
            continue

        distinct_item_set.add(item['asin'])

        if item['asin'] not in product_reviews:
            product_reviews[item['asin']] = "This item has no reviews"
            no_reviews_item_count += 1
    print('{0} items have no reviews(either in eval or test data)'.format(no_reviews_item_count))
    ##############################################################
    print('len of user_reviews_source: ' + str(len(user_reviews_source)))
    print('len of user_reviews_target: ' + str(len(user_reviews_target)))
    print('len of product: ' + str(len(product_reviews)))
    print('len of product in eval and test: ' + str(len(distinct_item_set)))

    concated = []
    for u in user_reviews_source:
        for rev in user_reviews_source[u]:
            concated.append(rev)

    num_users_source = len(concated)
    print('# of users in source: ' + str(num_users_source))

    for u in user_reviews_target:
        for rev in user_reviews_target[u]:
            concated.append(rev)
    num_users_target = num_users_source - len(concated)
    print('# of users in target: ' + str(num_users_target))

    for p in product_reviews:
        concated.append(product_reviews[p])

    for us in unlabeled_source:
        concated.append(unlabeled_source[us])

    for ut in unlabeled_target:
        concated.append(unlabeled_target[ut])

    tokenized_texts, word2idx, max_len = utils.tokenize(concated)
    all_input_ids = utils.encode(tokenized_texts, word2idx, max_len)

    tokenized_users_source = {}
    tokenized_users_target = {}
    tokenized_products = {}

    tokenized_unlabeled_source = {}
    tokenized_unlabeled_target = {}

    count = 0

    for u in user_reviews_source:

        tokenized_users_source[u] = [all_input_ids[idx] for idx in range(count, count + num_of_augmentation + 1)]
        count += num_of_augmentation + 1

    for u in user_reviews_target:
        tokenized_users_target[u] = [all_input_ids[idx] for idx in range(count, count + num_of_augmentation + 1)]
        count += num_of_augmentation + 1

    for p in product_reviews:
        tokenized_products[p] = all_input_ids[count]
        count += 1

    for us in unlabeled_source:
        tokenized_unlabeled_source[us] = all_input_ids[count]
        count += 1

    for ut in unlabeled_target:
        tokenized_unlabeled_target[ut] = all_input_ids[count]
        count += 1

    print('len of unlabeled source: ' + str(len(tokenized_unlabeled_source)))
    print('len of unlabeled target: ' + str(len(tokenized_unlabeled_target)))

    print('building datasets')

    train_data = models.Ratings_Dataset(train_user_item_ratings_source + train_user_item_ratings_target, tokenized_users_source,
                           tokenized_users_target, tokenized_products, isTrain=True, u_reviews_source=user_reviews_source, u_reviews_target=user_reviews_target, p_reviews=product_reviews)
    eval_data = models.Ratings_Dataset(eval_user_item_ratings, tokenized_users_source, tokenized_users_target, tokenized_products, u_reviews_source=user_reviews_source, u_reviews_target=user_reviews_target, p_reviews=product_reviews)
    test_data = models.Ratings_Dataset(test_user_item_ratings, tokenized_users_source, tokenized_users_target, tokenized_products, u_reviews_source=user_reviews_source, u_reviews_target=user_reviews_target, p_reviews=product_reviews)
    unlabeled_source_data = models.Unlabeled_Dataset(tokenized_unlabeled_source, domain_label=0)
    unlabeled_target_data = models.Unlabeled_Dataset(tokenized_unlabeled_target, domain_label=1)

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
    eval_dataloader = DataLoader(eval_data, batch_size=bs, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False)

    unlabled_source_dataloader = DataLoader(unlabeled_source_data, batch_size=bs//2, shuffle=True)
    unlabled_target_dataloader = DataLoader(unlabeled_target_data, batch_size=bs//2, shuffle=True)

    total = 0
    for it in product_reviews:
        rv = product_reviews[it]
        total += len(rv.split())
    print('avg item reviews : ' + str(total/len(product_reviews)) + ', total num of items : ' + str(len(product_reviews)))

    return train_dataloader, eval_dataloader, test_dataloader, unlabled_source_dataloader, unlabled_target_dataloader, word2idx