#-*- coding: utf-8 -*-
import os
import sys
import numpy as np
import sklearn.model_selection as ms

MAX_SEQ_LENGTH = 100
#MAX_SEQ_LENGTH = 128

def smiles2vec(filename, binary = False, repeat = False):

	# 0 으로 padding 하기 위한 내부 함수
	def pad(l, length, value = 0):

		return l + [0 for _ in range(length - len(l))]

	x_train = list()
	x_test = list()
	y_train = list()
	y_test = list()

	smiles = list()
        file_code = list()
        inversion_sym = list() #classification
	density = list() #regression

	# 파일 읽기
	with open(filename, 'r') as fs:

		lines = fs.readlines()

	# 각 줄을 처리, 유효한 줄만 사용
	for line in lines:

		col = list(map(str.strip, line.split(',')))

		# 원본 코드에서는 공백을 포함한 상태로 parsing 함
		# 여기서는 공백 없이 parsing 했기 때문에 MAX_SEQ_LENGTH - 2 로 조건을 주어야 같은 결과를 얻음
		if len(col) > 9 and len(col[2]) < MAX_SEQ_LENGTH - 2:

			# smiles 는 대소문자를 구별
			# space_group 은 대소문자 구별이 없음, 소문자화
			smiles.append(col[2])
                        file_code.append(col[1])
			density.append(col[10])
			inversion_sym.append(col[6])

	# assert 를 쓰면 프로그래밍 고수처럼 보임
	assert len(smiles) == len(inversion_sym) == len(density)

	# char2id 와 id2char 에서 사용하는 id 값은 1 부터 시작 (zero padding 때문에)
	# space2id 와 id2space 에서 사용하는 id 값은 0 부터 시작
	vocabulary = list(set(''.join(smiles)))
	groups = list(set(inversion_sym))
	char2id = {v:k + 1 for k, v in enumerate(vocabulary)}
	id2char = {v:k for k, v in char2id.items()}
	space2id = {v:k for k, v in enumerate(groups)}
	id2space = {v:k for k, v in space2id.items()}
	# 등장하는 character 와 inversion_sym 개수
	char_count = {v:''.join(smiles).count(v) for k, v in enumerate(vocabulary)}
	space_count = {v:inversion_sym.count(v) for k, v in enumerate(set(inversion_sym))}
	# smiles 를 character id 의 vector 로 변환, 가변 길이
	smiles_vec = np.array([pad(list(map(lambda s: char2id[s], smi)), MAX_SEQ_LENGTH) for smi in smiles])

	# 데이터를 클래스별로 나눔
	smiles_group = [list() for _ in range(len(groups))]

	for smi, sg in zip(smiles_vec, inversion_sym):

		smiles_group[space2id[sg]].append(smi)

	# 데이터를 분할할 때, 사용할 시드를 고정
	seed = np.random.randint(16384, size = 1)[0]

	# 데이터를 train : test = 9 : 1 로 분할
	for idx, val in enumerate(smiles_group):

		# binary 이면, 0 : monoclinic (가장 개수가 많음), 1 : 나머지
		# binary 가 아니면, 그냥 multiclass
		if binary:

			#split = ms.train_test_split(val, [0 if len(val) == max(space_count.values()) else 1] * len(val),
			#							test_size = 0.1, shuffle = True, random_state = seed)
			split = ms.train_test_split(val, [0 if len(val) == max(space_count.values()) else 1] * len(val),
										test_size = 0.1, shuffle = True, random_state = 0)

		else:

			#split = ms.train_test_split(val, [idx] * len(val), test_size = 0.1, shuffle = True, random_state = seed)
			split = ms.train_test_split(val, [idx] * len(val), test_size = 0.1, shuffle = True, random_state = 0)

		# 필요하면 클래스간의 데이터 개수를 같게 함
		# train set 에 대해서만 복제, test set 은 그대로
		rep = int(max(space_count.values()) / len(val)) if repeat else 1

		x_train += split[0] * rep
		x_test += split[1]
		y_train += split[2] * rep
		y_test += split[3]

	return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def test_run(binary = False, repeat = False):

	x_train, x_test, y_train, y_test = smiles2vec('CIF_to_labels.txt', binary = binary, repeat = repeat)
        np.save("X_tr.npy", x_train)
        np.save("Y_tr.npy", y_train)
	unique_elem_tr, counts_elem_tr = np.unique(y_train, return_counts = True)
	unique_elem, counts_elem = np.unique(y_test, return_counts = True)
	print('smiles2vec test')
	print('binary = {}, repeat = {}'.format(binary, repeat))
	print(np.asarray((unique_elem_tr, counts_elem_tr)))
	print(np.asarray((unique_elem, counts_elem)))
	print('')

# # 모듈 테스트용
test_run(False, False)
test_run(False, True)
test_run(True, False)
test_run(True, True)
