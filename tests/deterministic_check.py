import filecmp
import subprocess
from pathlib import Path


def run_once(out_diag):
    ds = Path('tests/fixtures/tiny_dataset.pt')
    cmd = [
        'python3','-m','rl.train',
        '--dataset', str(ds),
        '--epochs','5',
        '--steps-per-epoch','3',
        '--batch-size','2',
        '--lr','1e-3',
        '--device','cpu',
        '--diagnostics-out', str(out_diag),
        '--num-workers','0',
        '--seed','42'
    ]
    subprocess.run(cmd, check=True)


def main():
    a = Path('logs/det_a.json')
    b = Path('logs/det_b.json')
    if a.exists():
        a.unlink()
    if b.exists():
        b.unlink()
    run_once(a)
    run_once(b)
    same = filecmp.cmp(a, b, shallow=False)
    print('DETERMINISTIC ?', same)

if __name__ == '__main__':
    main()
