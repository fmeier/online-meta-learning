import os
import numpy as np
import torch
import progressbar
from torch.autograd import Variable

from meta_learning import utils as _utils
#from tf_utils import utils as _tf_utils

def compute_class_rate(outputs, labels):
    tmp, pred = torch.max(outputs, 1)
    return (pred.data == labels.data).float().mean()

def compute_test_class_rate(dataset, model, cuda):
    test_data = dataset.get_next_batch()
    test_inputs = test_data['x']
    test_labels = test_data['y']
    if cuda:
        test_inputs = test_inputs.cuda()
        test_labels = test_labels.cuda()

    outputs = model(Variable(test_inputs)) 
    return compute_class_rate(outputs, Variable(test_labels))

def train(model, optimizer, criterion, dataset, train_params, test_data=None):
    max_iter = train_params['max_iter']
    seed = train_params['seed']
    class_rate = False 

    #_utils.touch_pytorch(train_params['logdir'])
    torch.manual_seed(seed)
    loss = np.zeros(max_iter)
    if train_params.has_key('class_rate') and train_params['class_rate']:
        class_rate_train = np.zeros(max_iter)
        class_rate_test = np.zeros(max_iter)
        class_rate = True
    
    bar = progressbar.ProgressBar(max_value=max_iter)
    for n in range(max_iter):  # loop over the dataset multiple times

        data_batch = dataset.get_next_batch()
        if train_params['cuda']:
            data_batch['x'] = data_batch['x'].cuda()
            data_batch['y'] = data_batch['y'].cuda()

        inputs = Variable(data_batch['x'])
        targets = Variable(data_batch['y'])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        l = criterion(outputs, targets)
        # print('loss {} {}'.format(n, l.data[0]))
        # print('outputs {} targets {}'.format(outputs.data[0], targets.data[0]))
        # print('x1 {} x2 {}'.format(model.x1.data[0], model.x2.data[0]))
        loss[n] = l.data[0]
        if class_rate:
            class_rate_train[n] = compute_class_rate(outputs, targets)
            if test_data is not None:
               class_rate_test[n] = compute_test_class_rate(test_data,
                                                            model,
                                                            train_params['cuda'])
        l.backward()
        optimizer.step()

        if train_params['debug'] and not train_params['cuda']:
            if hasattr(model, 'forward_pass_data'):
                file_path = os.path.join(train_params['logdir'],
                                         "forward_pass_{}.pkl".format(n))
                #_tf_utils.pkl_save(file_path, model.forward_pass_data)

            file_path = os.path.join(train_params['logdir'],
                                     "loss_{}.pkl".format(n))
            #_tf_utils.pkl_save(file_path, {'loss': loss[n]})

        bar.update(n)

    print('Finished Training')
    res = {}
    res['loss'] = loss
    if class_rate:
        res['class_rate_train'] = class_rate_train
        res['class_rate_test'] = class_rate_test
        
    return res 
