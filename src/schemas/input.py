INPUT_SCHEMA = {
    'model': {
        'type': str,
        'required': False,
        'default': 'wangqixun/YamerMIX_v8'
    },
    'face_image': {
        'type': str,
        'required': True,
    },
    'pose_image': {
        'type': str,
        'required': False,
        'default': None
    },
    'prompt': {
        'type': str,
        'required': False,
        'default': 'a person'
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'style_name': {
        'type': str,
        'required': False,
        'default': 'Watercolor'
    },
    'num_steps': {
        'type': int,
        'required': False,
        'default': 30
    },
    'identitynet_strength_ratio': {
        'type': float,
        'required': False,
        'default': 0.8
    },
    'adapter_strength_ratio': {
        'type': float,
        'required': False,
        'default': 0.8
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 5
    },
    'seed': {
        'type': int,
        'required': False,
        'default': 42
    },
    'width': {
        'type': int,
        'required': False,
        'default': 0
    },
    'height': {
        'type': int,
        'required': False,
        'default': 0
    }
}
