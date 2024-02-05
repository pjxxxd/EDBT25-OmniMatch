import json
import random
from tqdm import tqdm


def data_conversion():
    with open('./reviews_CDs_and_Vinyl_5.json') as f:
        lines = f.readlines()

    print('data reading done')

    whole_target_data = []
    for line in lines:
        record = json.loads(line)
        if 'reviewText' in record and 'asin' in record and 'reviewerID' in record and 'overall' in record:
            whole_target_data.append(record)

    print('start converting')

    converted_data = {}

    for record in tqdm(whole_target_data):
        user_id = record['reviewerID']
        product_id = record['asin']
        # reviewText or summary
        review = record['reviewText']

        if user_id not in converted_data:
            converted_data[user_id] = {}
            converted_data[user_id][product_id] = review
        else:
            converted_data[user_id][product_id] = review

    print('start writing')

    with open('converted_music_reviewText.json', 'w', encoding='utf8') as json_file:
        json.dump(converted_data, json_file, ensure_ascii=False)

def generate_target_aux_doc(cold_start_users, domain='Books_Movies', mode='summary'):

    aux_doc_dict = {}

    if domain == 'Books_Movies':
        if mode == 'summary':
            converted_file_name = './converted_movies.json'
        elif mode == 'reviewText':
            converted_file_name = './converted_movies_reviewText.json'

        target_file_name = './aux_data/product_ratings_movies.json'
        source_file_name = './aux_data/product_ratings_book.json'

    elif domain == 'Movies_Music':
        if mode == 'summary':
            converted_file_name = './converted_music.json'
        elif mode == 'reviewText':
            converted_file_name = './converted_music_reviewText.json'

        target_file_name = './aux_data/product_ratings_music.json'
        source_file_name = './aux_data/product_ratings_movies.json'

    elif domain == 'Books_Music':
        if mode == 'summary':
            converted_file_name = './converted_music.json'
        elif mode == 'reviewText':
            converted_file_name = './converted_music_reviewText.json'

        target_file_name = './aux_data/product_ratings_music.json'
        source_file_name = './aux_data/product_ratings_book.json'

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
        # print('current user: ' + user)
        for products in source_users_to_products[user]:
            product = products[0]
            rating = products[1]
            # print('user records in source domain: ' + str(products))
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

            # for record in whole_target_data:
            #     if record['reviewerID'] == random_user and record['asin'] == random_review[0]:
            #         user_aux_doc += record[mode] + ' '

        aux_doc_dict[user] = user_aux_doc
        # print('{0} users rated this item in the source domain have reviews in target domain'.format(common_count))
        # print(count)
        # print(user_aux_doc)
    return aux_doc_dict

def generate_target_aux_doc_topk(cold_start_users, domain='Movies_Music', mode='summary',top_k=2):

    aux_doc_dict = {}

    if domain == 'Books_Movies':
        if mode == 'summary':
            converted_file_name = './converted_movies.json'
        elif mode == 'reviewText':
            converted_file_name = './converted_movies_reviewText.json'

        target_file_name = './aux_data/product_ratings_movies.json'
        source_file_name = './aux_data/product_ratings_book.json'

    elif domain == 'Movies_Music':
        if mode == 'summary':
            converted_file_name = './converted_music.json'
        elif mode == 'reviewText':
            converted_file_name = './converted_music_reviewText.json'

        target_file_name = './aux_data/product_ratings_music.json'
        source_file_name = './aux_data/product_ratings_movies.json'

    elif domain == 'Books_Music':
        if mode == 'summary':
            converted_file_name = './converted_music.json'
        elif mode == 'reviewText':
            converted_file_name = './converted_music_reviewText.json'

        target_file_name = './aux_data/product_ratings_music.json'
        source_file_name = './aux_data/product_ratings_book.json'

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
        # print('current user: ' + user)

        top_k_like_minded_users = {}

        for products in source_users_to_products[user]:
            product = products[0]
            rating = products[1]
            # print('user records in source domain: ' + str(products))
            users_with_reviews_source = source_data[product]

            like_minded_users = []
            like_minded_users_target = []

            for pair in users_with_reviews_source:

                if pair[0] in target_users_to_products:
                    common_count += 1

                if pair[1] == rating:
                    like_minded_users.append(pair[0])

            # print("product: " + product + ", has " + str(len(like_minded_users)) + " like minded users")

            for u in like_minded_users:
                if u in target_users_to_products:
                    count += 1
                    like_minded_users_target.append(u)

            # print(str(len(like_minded_users_target)) + " of them has reviews in target domain")

            # remove all like minded users in the cold start users set
            idx = 0
            while idx < len(like_minded_users_target):
                if like_minded_users_target[idx] in cold_start_users:
                    like_minded_users_target.pop(idx)
                else:
                    idx += 1

            if len(like_minded_users_target) == 0:
                continue

            # add to topK dict
            for u in like_minded_users_target:
                if u not in top_k_like_minded_users:
                    top_k_like_minded_users[u] = 1
                else:
                    top_k_like_minded_users[u] += 1

        max_count = 0
        for u in top_k_like_minded_users:
            if top_k_like_minded_users[u] > max_count:
                max_count = top_k_like_minded_users[u]
        # print("current user's most like-minded user share " + str(max_count) + " ratings")

        top_k_count = 0

        current_count = max_count

        while top_k > top_k_count:
            top_k_users_list = []
            for u in top_k_like_minded_users:
                if top_k_like_minded_users[u] == current_count:
                    top_k_users_list.append(u)

            if len(top_k_users_list) >= top_k - top_k_count:
                for x in range(top_k-top_k_count):
                    random_user = random.choice(top_k_users_list)
                    top_k_users_list.pop(top_k_users_list.index(random_user))

                    for product in converted_data[random_user]:
                        user_aux_doc += converted_data[random_user][product] + " "

                top_k_count += top_k - top_k_count
            else:
                for u in top_k_users_list:
                    for product in converted_data[u]:
                        user_aux_doc += converted_data[u][product] + " "

                top_k_count += len(top_k_users_list)
                current_count -= 1

        aux_doc_dict[user] = user_aux_doc
        # print(user_aux_doc)
        # print('***********')

    return aux_doc_dict

# generate_target_aux_doc_topk()
# data_conversion()