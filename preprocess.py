import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

rate_f = np.loadtxt('datasets/ratings_data.txt', dtype = np.int32)
trust_f = np.loadtxt('datasets/trust_data.txt', dtype = np.int32)

rate_list = []
trust_list = []

u_items_list = []  #存储每个用户交互过的物品iid和对应的评分，没有则为[(0, 0)]
#Store the item iid and the corresponding rating for each user interaction, or [(0, 0)] if none
u_users_list = []  #存储与每个物品相关联的用户及其评分，没有则为[(0, 0)]
#Store the users associated with each item and their ratings, or [(0, 0)] if none
u_users_items_list = [] #存储用户每个朋友的物品iid列表
#Store a list of items iid for each friend of the user
i_users_list = [] #存储与每个物品相关联的用户及其评分，没有则为[(0, 0)]
#Store the users associated with each item and their ratings, or [(0, 0)] if none

user_count = 0
item_count = 0
rate_count = 0

for s in rate_f:
	uid = s[0]
	iid = s[1]
	label = s[2]

	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid
	if label > rate_count:
		rate_count = label
	rate_list.append([uid, iid, label])

pos_list = []
for i in range(len(rate_list)):
	pos_list.append((rate_list[i][0], rate_list[i][1], rate_list[i][2]))

pos_list = list(set(pos_list))

random.shuffle(pos_list)
num_test=1000
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2 * num_test]
train_set = pos_list[2 * num_test:]
print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))

with open('datasets/dataset.pkl', 'wb') as f:
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
valid_df = pd.DataFrame(valid_set, columns = ['uid', 'iid', 'label'])
test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

click_df = pd.DataFrame(rate_list, columns = ['uid', 'iid', 'label'])
train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')

"""
u_items_list: 存储每个用户交互过的物品iid和对应的评分，没有则为[(0, 0)]
"""
for u in tqdm(range(user_count + 1)):
	hist = train_df[train_df['uid'] == u]
	u_items = hist['iid'].tolist()
	u_ratings = hist['label'].tolist()
	if u_items == []:
		u_items_list.append([(0, 0)])
	else:
		u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])

train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')

"""
i_users_list: 存储与每个物品相关联的用户及其评分，没有则为[(0, 0)]
"""
for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['iid'] == i]
	i_users = hist['uid'].tolist()
	i_ratings = hist['label'].tolist()
	if i_users == []:
		i_users_list.append([(0, 0)])
	else:
		i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])

for s in trust_f:
	uid = s[0]
	fid = s[1]
	if uid > user_count or fid > user_count:
		continue
	trust_list.append([uid, fid])

trust_df = pd.DataFrame(trust_list, columns = ['uid', 'fid'])
trust_df = trust_df.sort_values(axis = 0, ascending = True, by = 'uid')



"""
u_users_list: 存储每个用户互动过的用户uid；
u_users_items_list: 存储用户每个朋友的物品iid列表
"""
for u in tqdm(range(user_count + 1)):
	hist = trust_df[trust_df['uid'] == u]
	u_users = hist['fid'].unique().tolist()
	if u_users == []:
		u_users_list.append([0])
		u_users_items_list.append([[(0,0)]])
	else:
		u_users_list.append(u_users)
		uu_items = []
		for uid in u_users:
			uu_items.append(u_items_list[uid])
		u_users_items_list.append(uu_items)

with open('datasets/list.pkl', 'wb') as f:
	pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_users_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)


