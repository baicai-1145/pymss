import os, subprocess
root = '/content/s2'
subprocess.run(['shred', '-u', f'{root}/.hftok'], check=False) if os.path.exists(f'{root}/.hftok') else None
if os.path.exists(f'{root}/queue.pid') and os.path.exists(f'/proc/{open(f"{root}/queue.pid").read().strip()}'):
    print('queue already running pid', open(f'{root}/queue.pid').read().strip())
else:
    log = open(f'{root}/queue.log', 'a')
    p = subprocess.Popen(['bash', f'{root}/queue.sh'], cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    open(f'{root}/queue.pid', 'w').write(str(p.pid))
    print('launched queue pid', p.pid)
from huggingface_hub import get_token
print('hf token cached on VM:', bool(get_token()), '| .hftok removed:', not os.path.exists(f'{root}/.hftok'))
