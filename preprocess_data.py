import json
import gzip
import os

def read_datasets():
  movies_lines = []
  with gzip.open('reviews_Movies_and_TV_5.json.gz', "rb") as f:
    movies_lines = f.readlines()
    for idx in range(len(movies_lines)):
      movies_lines[idx] = json.loads(movies_lines[idx])
  print('domain Movies reading done')

  music_lines = []
  with gzip.open('reviews_CDs_and_Vinyl_5.json.gz', "rb") as f:
    music_lines = f.readlines()
    for idx in range(len(music_lines)):
      music_lines[idx] = json.loads(music_lines[idx])

  print('domain Music reading done')

  books_lines = []
  with gzip.open('reviews_Books_5.json.gz', "rb") as f:
    books_lines = f.readlines()
    for idx in range(len(books_lines)):
      books_lines[idx] = json.loads(books_lines[idx])

  print('domain Music reading done')

  datasets = {'Movies': movies_lines, 'Music': music_lines, 'Books': books_lines}
  return datasets

def preprocess_two_domains(data, domain_name_1, domain_name_2):
    print('------- Processing -------')
    print('domain {0} and domain {1}'.format(domain_name_1, domain_name_2))
    print('------- Processing -------')

    domain_1 = data[domain_name_1]
    domain_2 = data[domain_name_2]
    user_1 = set()
    for item in domain_1:
        user_1.add(item['reviewerID'])

    user_2 = set()
    for item in domain_2:
        user_2.add(item['reviewerID'])

    print('# of users in domain {0} :{1}, domain {2}: {3}'.format(domain_name_1, len(user_1), domain_name_2, len(user_2)))
    print('# of records in domain {0} :{1}, domain {2}: {3}'.format(domain_name_1, len(domain_1), domain_name_2, len(domain_2)))

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

    print('# of overlapping records in domain {0} :{1}, domain {2}: {3}'.format(domain_name_1, count_1, domain_name_2, count_2))

    output_file_name_1 = './data/' + domain_name_1 + domain_name_2 + '_' + domain_name_1 + '_raw.json'
    output_file_name_2 = './data/' + domain_name_1 + domain_name_2 + '_' + domain_name_2 + '_raw.json'

    if not os.path.isdir('data'):
        os.mkdir('data')

    with open(output_file_name_1, 'w', encoding='utf8') as json_file:
        json.dump(overlapping_domain_1, json_file, ensure_ascii=False)

    with open(output_file_name_2, 'w', encoding='utf8') as json_file:
        json.dump(overlapping_domain_2, json_file, ensure_ascii=False)

def prepare_unlabeled_data(datasets, domain_name):
    print('------- Processing -------')
    print('Preparing unlabeled data for domain {0}'.format(domain_name))
    print('------- Processing -------')

    data = datasets[domain_name]

    product = {}
    for item in data:
        if item['asin'] in product:
            product[item['asin']] += 1
        else:
            product[item['asin']] = 1

    # print('read product dictionary done')

    for k in list(product):
        if product[k] < 30:
            product.pop(k)

    count = 0
    while count < len(data):
        if data[count]['asin'] not in product:
            data.pop(count)
        else:
            count += 1

    # print('remove products done')

    reviewer = {}
    for item in data:
        if item['reviewerID'] not in reviewer:
            reviewer[item['reviewerID']] = 1
        else:
            reviewer[item['reviewerID']] += 1

    for k in list(reviewer):
        if reviewer[k] < 10:
            reviewer.pop(k)

    # print('poping data done')

    new_data = []
    for idx in range(len(data)):
        if data[idx]['reviewerID'] in reviewer and data[idx]['asin'] in product:
            new_data.append(data[idx])

    output_file_name_1 = './data/' + domain_name + '_unlabeled.json'

    if not os.path.isdir('data'):
        os.mkdir('data')

    with open(output_file_name_1, 'w', encoding='utf8') as json_file:
        json.dump(new_data, json_file, ensure_ascii=False)

    # print(len(new_data))

    # print('numebr of users: ' + str(len(reviewer)))
    # print('numebr of products: ' + str(len(product)))

def main():
  data = read_datasets()
  preprocess_two_domains(data, 'Movies', 'Music')
  preprocess_two_domains(data, 'Movies', 'Books')
  preprocess_two_domains(data, 'Books', 'Music')
  preprocess_two_domains(data, 'Books', 'Movies')
  preprocess_two_domains(data, 'Music', 'Movies')
  preprocess_two_domains(data, 'Music', 'Books')
  prepare_unlabeled_data(data, 'Books')
  prepare_unlabeled_data(data, 'Music')
  prepare_unlabeled_data(data, 'Movies')

if __name__ == '__main__':
    main()

