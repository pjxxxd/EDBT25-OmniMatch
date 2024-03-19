import json
import random
from tqdm import tqdm


def generate_target_aux_doc(cold_start_users, domain='Books_Movies', mode='summary'):

    aux_doc_dict = {}

    domain_index = domain.index("_")
    domain_1 = domain[:domain_index]
    domain_2 = domain[domain_index + 1:]

    if mode == 'summary':
        converted_file_name = './data/converted_' + domain_2 + '_summary.json'
    else:
        converted_file_name = './data/converted_' + domain_2 + '_reviewText.json'

    source_file_name = './data/product_ratings_' + domain_1 + '.json'
    target_file_name = './data/product_ratings_' + domain_2 + '.json'

    with open(converted_file_name) as json_file:
        converted_data = json.load(json_file)

    with open(target_file_name) as json_file:
        target_data = json.load(json_file)

    with open(source_file_name) as json_file:
        source_data = json.load(json_file)

    target_users_to_products = {}
    source_users_to_products = {}

    for product in source_data:
        for ratings in source_data[product]:
            user = ratings[0]
            rating = ratings[1]

            if ratings[0] not in source_users_to_products:
                source_users_to_products[user] = [[product, rating]]
            else:
                source_users_to_products[user].append([product, rating])

    for product in target_data:
        for ratings in target_data[product]:
            user = ratings[0]
            rating = ratings[1]

            if ratings[0] not in target_users_to_products:
                target_users_to_products[user] = [[product, rating]]
            else:
                target_users_to_products[user].append([product, rating])

    print('# of cold-start users: ' + str(len(cold_start_users)))

    for user in tqdm(cold_start_users):
        count = 0
        common_count = 0
        user_aux_doc = ""

        for products in source_users_to_products[user]:
            product = products[0]
            rating = products[1]

            users_with_reviews_source = source_data[product]

            like_minded_users = []
            like_minded_users_target = []

            for pair in users_with_reviews_source:

                if pair[0] in target_users_to_products:
                    common_count += 1

                if pair[1] == rating:
                    like_minded_users.append(pair[0])

            for u in like_minded_users:
                if u in target_users_to_products:
                    count += 1
                    like_minded_users_target.append(u)

            # remove all like minded users in the cold start users set
            idx = 0
            while idx < len(like_minded_users_target):
                if like_minded_users_target[idx] in cold_start_users:
                    like_minded_users_target.pop(idx)
                else:
                    idx += 1

            if len(like_minded_users_target) == 0:
                continue

            random_user = random.choice(like_minded_users_target)
            random_user_reviews = target_users_to_products[random_user]
            random_review = random.choice(random_user_reviews)

            user_aux_doc += converted_data[random_user][random_review[0]] + ' '

        aux_doc_dict[user] = user_aux_doc
    return aux_doc_dict