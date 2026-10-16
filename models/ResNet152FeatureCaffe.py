from models.ResNetFeature import *
from utils import *
from os import path
        
def create_model(use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, caffe=False, log_dir=None, test=False):
    
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=dropout)
    
    if not test:

        #assert(caffe != stage1_weights)

        if caffe:
            print('Loading Caffe Pretrained ResNet 152 Weights.')
            resnet152 = init_weights(model=resnet152,
                                     weights_path='./pretrain/caffe_resnet152.pth',
                                     caffe=True)
        elif stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 152 Weights.' % dataset)
            if log_dir is not None:
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading weights from %s' % weight_dir)
            resnet152 = init_weights(model=resnet152,
                                     weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'))
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet152
