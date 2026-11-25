"""Watch the Colab S2 evaluation file and reconnect on transport failure."""
import os
import shlex
import subprocess
from pathlib import Path

REMOTE = r'''
import json
import os
import re
import select
import subprocess
from pathlib import Path

os.chdir('/content/s2')
result = Path('out/S2/eval-50000.json')
watch = subprocess.Popen(
    ['tail', '-c', '+1', '-F', str(result), 'queue.log'],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)
try:
    while True:
        ready, _, _ = select.select([watch.stdout], [], [], 20)
        if not ready:
            print('S2_HEARTBEAT', flush=True)
            continue
        if not os.read(watch.stdout.fileno(), 65536):
            raise RuntimeError('tail exited')
        try:
            report = json.loads(result.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            report = None
        if report is not None and report['step'] == 50000:
            print('S2_RESULT', report['median_dB'], report['windows'], flush=True)
            break
        failure = re.search(r'^\[[^\]]+\] (?:STAGE|CHECK|PREPARE|IMPORT|PIP)_FAILED\b.*', Path('queue.log').read_text(), re.M)
        if failure:
            print('S2_ERROR', failure.group(), flush=True)
            break
finally:
    watch.terminate()
    watch.wait()
'''


def main():
    cli = str(Path.home() / '.local/bin/colab')
    base = [cli, '--auth', 'adc']
    env = dict(os.environ, GOOGLE_APPLICATION_CREDENTIALS=str(Path.home() / 'Downloads/fbsdev011_colab_only_adc.json'))
    for attempt in range(1, 9):
        with subprocess.Popen(
            [*base, 'ssh', '-s', 's2', '--', 'python3 -u -c ' + shlex.quote(REMOTE)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        ) as process:
            result = False
            for line in process.stdout:
                if line.startswith(('S2_RESULT ', 'S2_ERROR ')):
                    print(line.rstrip(), flush=True)
                    if line.startswith('S2_ERROR '):
                        return 1
                    result = True
            if result:
                return 0
        print(f'S2_WATCH_DISCONNECTED attempt={attempt}', flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
