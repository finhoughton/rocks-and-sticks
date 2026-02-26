import json
import subprocess
from pathlib import Path


def run_overfit():
    ds = Path('tests/fixtures/tiny_dataset.pt')
    out_model = Path('/tmp/overfit.pt')
    diag = Path('logs/overfit.json')
    if diag.exists():
        diag.unlink()
    cmd = [
        'python3','-m','rl.train',
        '--dataset', str(ds),
        '--epochs','200',
        '--steps-per-epoch','3',
        '--batch-size','2',
        '--lr','1e-3',
        '--device','cpu',
        '--out', str(out_model),
        '--diagnostics-out', str(diag),
        '--num-workers','0'
    ]
    subprocess.run(cmd, check=True)
    # check last few diagnostics for low loss
    lines = [l.strip() for l in diag.read_text(encoding='utf-8').splitlines() if l.strip()]
    last = json.loads(lines[-1])
    print('last loss', last.get('loss'))
    return float(last.get('loss'))

if __name__ == '__main__':
    loss = run_overfit()
    if loss < 0.2:
        print('OVERFIT OK')
    else:
        print('OVERFIT FAILED', loss)
