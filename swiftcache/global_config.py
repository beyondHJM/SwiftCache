global_config={
    'gpu_memory_utilization':0.97,
    'kv_cache_strategy':'external_gpu', # 可选: 'normal', 'cpu', 'external_gpu'
    'kvcache_block_size':256,
    'use_priority_queue': True,
    'num_kvcache_blocks':-1,
    # 'num_blocks_start_end':[0,300],
    'transfer_full_kvcache':False,
    'num_external_kvcache':-1,
    'os_pid_dir':'/tmp/'
}

assert global_config['kv_cache_strategy'] in ['normal', 'cpu', 'external_gpu']

if global_config['kv_cache_strategy'] == 'cpu':
    global_config['gpu_memory_utilization'] = 0.9
    # global_config['num_kvcache_blocks'] *= 5
elif global_config['kv_cache_strategy'] == 'normal':
    global_config['gpu_memory_utilization'] = 0.9
    global_config['num_blocks_start_end'] = [0]
    

elif global_config['kv_cache_strategy'] == 'external_gpu':
    global_config['gpu_memory_utilization'] = 0.9
    # global_config['num_kvcache_blocks'] = global_config['num_blocks_start_end'][-1]


