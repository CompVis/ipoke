poke_embedder_models ={
    'iper-ss128-bn64-endpoint10f-np5': {
        'ckpt': 'logs/poke_encoder/ckpt/iper_128/0/epoch=17-lpips-val=0.298.ckpt',
        'model_name': 'iper_128',
        'tgt_name': 'iper_128'
    },
    'h36m-ss128-bn64-endpoint10f-np5': {
        'ckpt': 'logs/poke_encoder/ckpt/h36m_128/0/epoch=19-lpips-val=0.109.ckpt',
        'model_name': 'h36m_128',
        'tgt_name' : 'h36m_128'
    },
    'plants-ss128-bn64-endpoint10f-np5': {
        'ckpt': 'logs/poke_encoder/ckpt/plants_128/0/epoch=79-lpips-val=0.301.ckpt',
        'model_name': 'plants_128',
        'tgt_name': 'plants_128'
    },
    'iper-ss64-bn8x8x64-endpoint10f-np5': {
        'ckpt': 'logs/poke_encoder/ckpt/iper_64/0/epoch=16-lpips-val=0.172.ckpt',
        'model_name': 'iper_64',
        'tgt_name': 'iper_64'
    },
    'taichi-ss128-bn8x8x64-endpoint10f-np5': {
        'ckpt': 'logs/poke_encoder/ckpt/taichi_128/0/epoch=31-lpips-val=0.314.ckpt',
        'model_name': 'taichi_128',
        'tgt_name': 'taichi_128'
    },
    'taichi-ss64-bn8x8x64-endpoint10f-np5': {
        'ckpt': 'logs/poke_encoder/ckpt/taichi_64/0/epoch=14-lpips-val=0.229.ckpt',
        'model_name': 'taichi_64',
        'tgt_name': 'taichi_64'
    },
    'plants-ss64-bn8x8x64-endpoint10f-np5': {
        'ckpt': 'logs/poke_encoder/ckpt/plants_64/0/epoch=60-lpips-val=0.183.ckpt',
        'model_name': 'plants_64',
        'tgt_name': 'plants_64'
    },
        'h36m-ss64-bn8x8x64-endpoint10f-np5': {
        'ckpt': 'logs/poke_encoder/ckpt/h36m_64/0/epoch=16-lpips-val=0.073.ckpt',
        'model_name': 'h36m_64'
    },
}
first_stage_models = {
'plants-ss128-bn64-mf10' : {
    'ckpt': 'logs/first_stage/ckpt/plants_128/0/epoch=17-FVD-val=65.191.ckpt',
    'model_name': 'plants_128',
    'tgt_name':'plants_128'
},
    'h36m-ss128-bn64-mf10' : {
        'ckpt': 'logs/first_stage/ckpt/h36m_128/0/epoch=13-FVD-val=109.079.ckpt',
        'model_name': 'h36m_128',
        'tgt_name':'h36m_128'
    },
'taichi-ss128-bn32-mf10': {
        'ckpt': 'logs/first_stage/ckpt/taichi_128/0/epoch=10-FVD-val=157.258.ckpt',
        'model_name': 'taichi_128',
        'tgt_name': 'taichi_128'
},
'plants-ss64-bn32-mf10': {
        'ckpt': 'logs/first_stage/ckpt/plants_64/0/epoch=18-FVD-val=61.761.ckpt',
        'model_name': 'plants_64',
    'tgt_name': 'plants_64'
},
'h36m-ss64-bn64-mf10': {
        'ckpt': 'logs/first_stage/ckpt/h36m_64/0/epoch=18-FVD-val=108.995.ckpt',
        'model_name': 'h36m_64'
},
'iper-ss64-bn32-mf10': {
        # run name is false here, model was trained with z_dim = 32, as indicated in the dict key
        'ckpt': 'logs/first_stage/ckpt/iper_64/0/epoch=28-FVD-val=67.734.ckpt',
        'model_name': 'iper_64',
        'tgt_name': 'iper_64'
},
'taichi-ss64-bn32-mf10': {
    # run name is false here, model was trained with z_dim = 32, as indicated in the dict key
    'ckpt': 'logs/first_stage/ckpt/taichi_64/0/epoch=20-FVD-val=113.079.ckpt',
        'model_name': 'taichi_64',
    'tgt_name': 'taichi_64'
},
'iper-ss128-bn32-mf10-complex': {
        # run name is false here, model was trained with z_dim = 32, as indicated in the dict key
        'ckpt': 'logs/first_stage/ckpt/iper_128/0/epoch=17-FVD-val=61.491.ckpt',
        'model_name': 'iper_128',
        'tgt_name': 'iper_128'
},
}
conditioner_models = {
'plants-ss128-bn64': {
        'ckpt': 'logs/img_encoder/ckpt/plants_128/0/epoch=71-lpips-val=0.051.ckpt',
        'model_name': 'plants_128',
        'tgt_name': 'plants_128'
    },
'iper-ss128-bn64': {
        'ckpt': 'logs/img_encoder/ckpt/iper_128/0/epoch=12-lpips-val=0.026.ckpt',
        'model_name': 'iper_128',
        'tgt_name': 'iper_128'
    },
'h36m-ss128-bn64': {
        'ckpt': 'logs/img_encoder/ckpt/h36m_128/0/epoch=12-lpips-val=0.067.ckpt',
        'model_name': 'h36m_128',
        'tgt_name': 'h36m_128'
    },
'plants-ss64-bn64': {
        'ckpt': 'logs/img_encoder/ckpt/plants_64/0/last.ckpt',
        'model_name': 'plants_64',
        'tgt_name': 'plants_64'
    },
'iper-ss64-bn64': {
        'ckpt': 'logs/img_encoder/ckpt/iper_64/0/last.ckpt',
        'model_name': 'iper_64',
        'tgt_name': 'iper_64'
    },
'h36m-ss64-bn64': {
        'ckpt': 'logs/img_encoder/ckpt/h36m_64/0/last.ckpt',
        'model_name': 'h36m_64'
    },
'taichi-ss128-bn64': {
        'ckpt': 'logs/img_encoder/ckpt/taichi_128/0/epoch=8-lpips-val=0.110.ckpt',
        'model_name': 'taichi_128',
        'tgt_name': 'taichi_128'
    },
'taichi-ss64-bn64': {
        'ckpt': 'logs/img_encoder/ckpt/taichi_64/0/epoch=14-lpips-val=0.006.ckpt',
        'model_name': 'taichi_64',
        'tgt_name': 'taichi_64'
    },
}

flow_conditioner_models ={}