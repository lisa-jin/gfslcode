from __future__ import print_function


from . import model_dict


def create_model(name, dataset='miniImageNet'):
    """create model by name"""
    if dataset == 'miniImageNet' or dataset == 'tieredImageNet':
        if name.endswith('v2') or name.endswith('v3'):
            model = model_dict[name]()
        elif name.startswith('resnet50'):
            print('use imagenet-style resnet50')
            model = model_dict[name]()
        elif name.startswith('resnet') or name.startswith('seresnet') or name.startswith('subresnet') :
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=5)
        elif name.startswith('wrn'):
            model = model_dict[name]()
        elif name.startswith('convnet'):
            model = model_dict[name]()
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    elif dataset == 'CIFAR-FS':
        if name.startswith('resnet') or name.startswith('seresnet') or name.startswith('subresnet') :
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=2)
        elif name.startswith('convnet'):
            model = model_dict[name]()
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))

    elif dataset == 'CUB' or dataset=='Dog':
        if name.endswith('v2') or name.endswith('v3'):
            model = model_dict[name]()
        elif name.startswith('resnet50'):
            print('use imagenet-style resnet50')
            model = model_dict[name]()
        elif name.startswith('resnet') or name.startswith('seresnet') or name.startswith('subresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=5)
        elif name.startswith('wrn'):
            model = model_dict[name]()
        elif name.startswith('convnet'):
            model = model_dict[name]()
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    return model


def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segments = model_path.split('/')[-2].split('_')
    if ':' in segments[0]:
        return segments[0].split(':')[-1]
    else:
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
