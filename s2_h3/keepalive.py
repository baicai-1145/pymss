import os, time
root = '/content/s2'; pos, hb = 0, 0
print('keepalive attached; queue pid', open(f'{root}/queue.pid').read().strip(), 'alive:', os.path.exists(f'/proc/{open(f"{root}/queue.pid").read().strip()}'), flush=True)
while True:
    with open(f'{root}/queue.log') as f:
        s = f.read()
    new = s[pos:]; pos = len(s)
    if new:
        print(new, end='', flush=True)
    if any(k in s for k in ('ALL_DONE', 'STAGE_FAILED', 'CHECK_FAILED', 'PREPARE_FAILED', 'IMPORT_FAILED')):
        break
    if not new:
        print(f'.{hb}', end='', flush=True); hb += 1
    time.sleep(20)
print('\nKEEPALIVE_EXIT', flush=True)
