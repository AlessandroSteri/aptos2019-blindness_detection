from time import time
import uuid
import os
def id_gen(unique=True, sequential=True):
    assert unique or sequential
    seq_id = ''
    unq_id = ''
    if sequential:
        seq_id = str(int(time()*100))[1:]
    if unique:
        unq_id = str(uuid.uuid4()).replace('-', '').upper()[0:2]
    return seq_id + '_' + unq_id


# check if directory exists, if not it makes it
def mkdir(directory, log=False):
    if not exists(directory):
        os.makedirs(directory)
        if log:
            print('Created new directory: {}'.format(directory))

def exists(path):
    return os.path.exists(path)

