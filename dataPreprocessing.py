import json
import random
import utils
import models
import aux_proc
from torch.utils.data import DataLoader


def CATN_one_domain():
    with open('./reviews_Books_5.json') as f:
        lines = f.readlines()

    data = []
    for l in lines:
        record = json.loads(l)
        if 'reviewText' in record and 'asin' in record and 'reviewerID' in record and 'overall' in record:
            data.append(record)
    print('data reading done')

    product = {}
    for item in data:
        if item['asin'] in product:
            product[item['asin']] += 1
        else:
            product[item['asin']] = 1

    print('review product dictionary done')

    for k in list(product):
        if product[k] < 30:
            product.pop(k)

    count = 0
    while count < len(data):
        if data[count]['asin'] not in product:
            data.pop(count)
        else:
            count += 1

    print('remove products done')

    reviewer = {}
    for item in data:
        if item['reviewerID'] not in reviewer:
            reviewer[item['reviewerID']] = 1
        else:
            reviewer[item['reviewerID']] += 1

    for k in list(reviewer):
        if reviewer[k] < 10:
            reviewer.pop(k)

    print('poping data done')

    new_data = []
    for idx in range(len(data)):
        if data[idx]['reviewerID'] in reviewer and data[idx]['asin'] in product:
            new_data.append(data[idx])

    print(len(new_data))

    print('numebr of users: ' + str(len(reviewer)))
    print('numebr of products: ' + str(len(product)))

    with open('Books_30_10.json', 'w', encoding='utf8') as json_file:
        json.dump(new_data, json_file, ensure_ascii=False)


def CATN_two_domains():
    with open('./Books_30_10.json') as json_file:
        domain_1 = json.load(json_file)

    with open('./Movies_30_10.json') as json_file:
        domain_2 = json.load(json_file)

    user_1 = set()
    for item in domain_1:
        user_1.add(item['reviewerID'])

    user_2 = set()
    for item in domain_2:
        user_2.add(item['reviewerID'])

    print('# of users in domain 1 :{0}, domain 2: {1}'.format(len(user_1), len(user_2)))
    print('# of records in domain 1 :{0}, records 2: {1}'.format(len(domain_1), len(domain_2)))

    overlapping_users = set()

    for user in user_1:
        if user in user_2:
            overlapping_users.add(user)

    print('# of overlapping users: ' + str(len(overlapping_users)))

    overlapping_domain_1 = {}
    for item in domain_1:
        if item['reviewerID'] in overlapping_users:
            if item['reviewerID'] not in overlapping_domain_1:
                overlapping_domain_1[item['reviewerID']] = [item]
            else:
                overlapping_domain_1[item['reviewerID']].append(item)

    overlapping_domain_2 = {}
    for item in domain_2:
        if item['reviewerID'] in overlapping_users:
            if item['reviewerID'] not in overlapping_domain_2:
                overlapping_domain_2[item['reviewerID']] = [item]
            else:
                overlapping_domain_2[item['reviewerID']].append(item)

    count_1 = 0
    for idx in overlapping_domain_1:
        count_1 += len(overlapping_domain_1[idx])

    count_2 = 0
    for idx in overlapping_domain_2:
        count_2 += len(overlapping_domain_2[idx])

    print('# of overlapping records in domain 1 :{0}, records 2: {1}'.format(count_1, count_2))

    with open('BooksMovies_Books.json', 'w', encoding='utf8') as json_file:
        json.dump(overlapping_domain_1, json_file, ensure_ascii=False)

    with open('BooksMovies_Movies.json', 'w', encoding='utf8') as json_file:
        json.dump(overlapping_domain_2, json_file, ensure_ascii=False)


def train_eval_test_split_per_domain(domains):
    ratio = 1
    if domains =="Movies_Music":
        domain1_file = './MoviesMusic_Movies_alcdr.json'
        domain2_file = './MoviesMusic_Music_alcdr.json'
        # domain1_file = './MoviesMusic_Movies.json'
        # domain2_file = './MoviesMusic_Music.json'
    elif domains == "Books_Movies":
        domain1_file = './BooksMovies_Books_alcdr.json'
        domain2_file = './BooksMovies_Movies_alcdr.json'
        # domain1_file = './BooksMovies_Books.json'
        # domain2_file = './BooksMovies_Movies.json'
    elif domains == "Books_Music":
        domain1_file = './BooksMusic_Books_alcdr.json'
        domain2_file = './BooksMusic_Music_alcdr.json'
        # domain1_file = './BooksMusic_Books.json'
        # domain2_file = './BooksMusic_Music.json'

    with open(domain1_file) as json_file:
        domain_1 = json.load(json_file)

    with open(domain2_file) as json_file:
        domain_2 = json.load(json_file)

    print(len(domain_1))

    users = []
    for u in domain_1:
        users.append(u)

    random.shuffle(users)

    train_len = int(0.8 * len(users))
    eval_len = int(0.1 * len(users))

    fraction_training = int(train_len * ratio)
    train_users = users[:fraction_training]
    eval_users = users[train_len:train_len + eval_len]
    test_users = users[train_len + eval_len:]
    cold_start_users = eval_users + test_users

    print('# of train, eval, test users : {0}, {1}, {2}'.format(len(train_users), len(eval_users), len(test_users)))

    training_records_source = []
    training_records_target = []

    # all the records in the source domain are used as training samples
    # for u in users:
    #     for record in domain_1[u]:
    #         training_records_source.append(record)

    #############
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
        domain1_file = './Movies_30_10.json'
        domain2_file = './Music_30_10.json'
    elif domains == "Books_Movies":
        domain1_file = './Books_30_10.json'
        domain2_file = './Movies_30_10.json'
    elif domains == "Books_Music":
        domain1_file = './Books_30_10.json'
        domain2_file = './Music_30_10.json'

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
    train_source_list, train_target_list, eval_list, test_list, cold_start_users= train_eval_test_split_per_domain(domain)

    # train_source_list, train_target_list, eval_list, test_list = train_eval_test_split_CATN_setup(domain)
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
    # with open('./Movies_Music_seed_0.json') as json_file:
    #     cold_start_target_dict = json.load(json_file)

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

    print('# of users + products: ' + str(len(concated)))

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

    # unlabeled_data = models.Unlabeled_Dataset(tokenized_unlabeled_source, tokenized_unlabeled_target)

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
    eval_dataloader = DataLoader(eval_data, batch_size=bs, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False)
    # unlabled_dataloader = DataLoader(unlabeled_data, batch_size=bs, shuffle=True)
    unlabled_source_dataloader = DataLoader(unlabeled_source_data, batch_size=bs//2, shuffle=True)
    unlabled_target_dataloader = DataLoader(unlabeled_target_data, batch_size=bs//2, shuffle=True)

    total = 0
    for it in product_reviews:
        rv = product_reviews[it]
        total += len(rv.split())
    print('avg item reviews : ' + str(total/len(product_reviews)) + ', total num of items : ' + str(len(product_reviews)))

    return train_dataloader, eval_dataloader, test_dataloader, unlabled_source_dataloader, unlabled_target_dataloader, word2idx