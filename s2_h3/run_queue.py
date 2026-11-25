import os, subprocess, time
root = '/content/s2'
if os.path.exists(f'{root}/queue.pid') and os.path.exists(f'/proc/{open(f"{root}/queue.pid").read().strip()}'):
    print('queue already running pid', open(f'{root}/queue.pid').read().strip(), flush=True)
    p = None
else:
    log = open(f'{root}/queue.log', 'a')
    p = subprocess.Popen(['bash', f'{root}/queue.sh'], cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    open(f'{root}/queue.pid', 'w').write(str(p.pid))
    print('launched queue pid', p.pid, flush=True)
# Keep this kernel cell BUSY (Colab keeps the session alive while the kernel is active): stream queue.log, heartbeat every 20 s.
pos, hb = 0, 0
while True:
    if p is not None and p.poll() is not None:
        break
    time.sleep(20)
    with open(f'{root}/queue.log') as f:
        f.seek(pos); chunk = f.read(); pos = f.tell()
    print(chunk if chunk else f'.{hb}', end='', flush=True); hb += 1
with open(f'{root}/queue.log') as f:
    f.seek(pos); print(f.read(), end='', flush=True)
print('\nQUEUE_EXIT', None if p is None else p.returncode(), flush=True)
