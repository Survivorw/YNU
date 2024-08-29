from __future__ import print_function
from keras_vggface.models import  VGG16
def VGGFace(include_top=True, model='vgg16', weights='vggface',
            input_tensor=None, input_shape=None,
            pooling=None,
            classes=None):
    

    if weights not in {'vggface', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `vggface`'
                         '(pre-training on VGGFace Datasets).')

    if model == 'vgg16':

        if classes is None:
            classes = 2622

        if weights == 'vggface' and include_top and classes != 2622:
            raise ValueError(
                'If using `weights` as vggface original with `include_top`'
                ' as true, `classes` should be 2622')

        return VGG16(include_top=include_top, input_tensor=input_tensor,
                     input_shape=input_shape, pooling=pooling,
                     weights=weights,
                     classes=classes)