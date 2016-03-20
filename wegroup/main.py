import utility as u

train_file = "data_train.txt"
test_file = "shuffle_data_test.txt"
data_folder = 'data'
columns = ['click', 'weekday', 'hour', 'timestamp', 'log_type', 'user_id', 'user_agent', 'ip', 'region', 'city',
           'ad_exchange', 'domain', 'url', 'anon_url_id', 'ad_slot_id', 'width', 'height', 'visibility', 'format',
           'price', 'creative_id', 'key_page_url', 'advertiser_id', 'user_tags']

train, test = u.import_tr_te(train_file, test_file, columns, data_folder)
print(train.describe())
print(test.describe())
