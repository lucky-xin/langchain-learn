from modelscope import snapshot_download

if __name__ == '__main__':
    # model_id = 'OpenNLPLab/TransNormerLLM-7B'
    model_id = 'deepseek-ai/DeepSeek-R1'
    nodel_dir = snapshot_download(model_id, cache_dir='/tmp/model')
