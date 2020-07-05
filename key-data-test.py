# -*- coding: utf-8 -*-

import os
import pickle
from collections import OrderedDict, Counter
import random
import glob

def main():
    random.seed(0)

    #valid_keys = glob.glob('../data/*')
    data_dir = os.path.join('.', 'data')
    file_pattern = os.path.join(data_dir, '*')

    valid_keys = glob.glob(file_pattern)
    valid_keys = [v.split('/')[-1] for v in valid_keys]

    print('valid_keys:len({})\n{}'.format(len(valid_keys), valid_keys))
    print('+' * 10)

    keys = [v.split('_')[0] for v in valid_keys]
    print('keys:len:{}, type:{}\n{}'.format(len(keys), type(keys), keys))

    key_set = set(keys)
    print('key_set:len:{}\n{}'.format(len(key_set), key_set))

    print('+' * 10)

    ordered_dict = OrderedDict.fromkeys(keys)
    print('ordered_dict:len:{}, type:{}\n{}'.format(len(ordered_dict), type(ordered_dict), ordered_dict))

    print('+' * 10)

    counter = Counter(keys)
    print('counter:len:{}, type:{}\n{}'.format(len(counter), type(counter), counter))

    print('+' * 10)

    #dude_gene = list(OrderedDict.fromkeys([v.split('_')[0] for v in valid_keys]))
    dude_gene = list(ordered_dict)
    print('dude_gene:len({})\n{}'.format(len(dude_gene), dude_gene))

    print('+' * 10)

    test_dude_gene = ['egfr', 'parp1', 'fnta', 'aa2ar', 'pygm', 'kith', 'met', 'abl1', 'ptn1', 'casp3', 'hdac8', 'grik1', 'kpcb', 'ada', 'pyrd', 'ace', 'aces', 'pgh1', 'aldr', 'kit', 'fa10', 'pa2ga', 'fgfr1', 'cp3a4', 'wee1', 'tgfr1']
    test_dude_gene_set = set(test_dude_gene)
    print('len(test_dude_gene):{} vs. len(test_dude_gene_set):{}'.format(len(test_dude_gene), len(test_dude_gene_set)))
    print('+' * 10)

    real_keys = [key for key in key_set            if key     in test_dude_gene_set]
    fake_keys = [key for key in test_dude_gene_set if key not in key_set]
    print('len(real_keys):{} vs. len(fake_keys):{}({})'.format(len(real_keys), len(fake_keys), fake_keys))
    print('+' * 10)

    test_dude_gene = [key for key in test_dude_gene if key in key_set]
    test_dude_gene_set = set(test_dude_gene)
    print('### --> len(test_dude_gene):{} vs. len(test_dude_gene_set):{}'.format(len(test_dude_gene), len(test_dude_gene_set)))
    print('+' * 10)

    train_dude_gene = [p for p in dude_gene if p not in test_dude_gene_set]
    train_dude_gene_set = set(train_dude_gene)
    print('len(train_dude_gene):{} vs. len(train_dude_gene_set):{}'.format(len(train_dude_gene), len(train_dude_gene_set)))
    print('+' * 10)

    print('len(train_dude_gene_set):{}, len(test_dude_gene_set):{}'.format(len(train_dude_gene_set), len(test_dude_gene_set)))
    print('+' * 10)

    #train_keys = [k for k in valid_keys if k.split('_')[0] in train_dude_gene]
    #print('#1:train_keys:len({})\n{}'.format(len(train_keys), train_keys))

    train_keys = [k for k in valid_keys if k.split('_')[0] in train_dude_gene_set]
    #print('#2:train_keys:len({})\n{}'.format(len(train_keys), train_keys))
    print('train_keys:len({})\n{}'.format(len(train_keys), train_keys))

    print('+' * 10)

    test_keys = [k for k in valid_keys if k.split('_')[0] in test_dude_gene_set]
    print('test_keys:len({})\n{}'.format(len(test_keys), test_keys))

    print('+' * 10)

    print ('Num train keys: ', len(train_keys))
    print ('Num test keys:  ', len(test_keys ))

    key_dir = os.path.join('.', 'keys')

    #with open('train_dude_gene.pkl', 'wb') as f:
    data_file_path = os.path.join(key_dir, 'train_dude_gene.pkl')
    with open(data_file_path, 'wb') as f:
        pickle.dump(train_dude_gene, f)
        pass

    #with open('test_dude_gene.pkl', 'wb') as f:
    data_file_path = os.path.join(key_dir, 'test_dude_gene.pkl')
    with open(data_file_path, 'wb') as f:
         pickle.dump(test_dude_gene, f)
         pass

    #with open('train_keys.pkl', 'wb') as f:
    data_file_path = os.path.join(key_dir, 'train_keys.pkl')
    with open(data_file_path, 'wb') as f:
         pickle.dump(train_keys, f)
         pass

    #with open('test_keys.pkl', 'wb') as f:
    data_file_path = os.path.join(key_dir, 'test_keys.pkl')
    with open(data_file_path, 'wb') as f:
         pickle.dump(test_keys, f)
         pass

    pass

if __name__ == '__main__':
    main()
    pass



#dude_gene =  list(OrderedDict.fromkeys([v.split('_')[0] for v in valid_keys]))
#test_dude_gene = ['egfr', 'parp1', 'fnta', 'aa2ar', 'pygm', 'kith', 'met', 'abl1', 'ptn1', 'casp3', 'hdac8', 'grik1', 'kpcb', 'ada', 'pyrd', 'ace', 'aces', 'pgh1', 'aldr', 'kit', 'fa10', 'pa2ga', 'fgfr1', 'cp3a4', 'wee1', 'tgfr1']
#train_dude_gene = [p for p in dude_gene if p not in test_dude_gene]
#
#train_keys = [k for k in valid_keys if k.split('_')[0] in train_dude_gene]
#test_keys = [k for k in valid_keys if k.split('_')[0] in test_dude_gene]
#
#print ('Num train keys: ', len(train_keys))
#print ('Num test keys: ', len(test_keys))
#
#with open('train_dude_gene.pkl', 'wb') as f:
#    pickle.dump(train_dude_gene, f)
#with open('test_dude_gene.pkl', 'wb') as f:
#    pickle.dump(test_dude_gene, f)
#with open('train_keys.pkl', 'wb') as f:
#    pickle.dump(train_keys, f)
#with open('test_keys.pkl', 'wb') as f:
#    pickle.dump(test_keys, f)
#
