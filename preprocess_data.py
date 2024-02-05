import json


def preprocess_two_domains():
    with open('./reviews_Movies_and_TV_5.json') as f:
        lines = f.readlines()

    domain_1 = []
    for l in lines:
        record = json.loads(l)
        if 'reviewText' in record and 'asin' in record and 'reviewerID' in record and 'overall' in record:
            domain_1.append(record)
    print('domain 1 reading done')

    with open('./reviews_CDs_and_Vinyl_5.json') as f:
        lines = f.readlines()

    domain_2 = []
    for l in lines:
        record = json.loads(l)
        if 'reviewText' in record and 'asin' in record and 'reviewerID' in record and 'overall' in record:
            domain_2.append(record)
    print('domain 2 reading done')

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

    #with open('BooksMovies_Books_alcdr.json', 'w', encoding='utf8') as json_file:
        #json.dump(overlapping_domain_1, json_file, ensure_ascii=False)

    #with open('BooksMovies_Movies_alcdr.json', 'w', encoding='utf8') as json_file:
        #json.dump(overlapping_domain_2, json_file, ensure_ascii=False)


preprocess_two_domains()