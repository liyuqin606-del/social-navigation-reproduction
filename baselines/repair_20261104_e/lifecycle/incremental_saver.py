#!/usr/bin/env python3
"""Verify only atomic, already collected units. No network or allocation."""
import argparse
from pathlib import Path
import time
from transport import read, save, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    a = p.parse_args()
    while True:
        path = a.run / 'COLLECTED.json'
        if path.exists():
            manifest = read(path)
            for row in manifest['files']:
                target = a.run / 'incremental_cache' / row['path']
                if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                    raise RuntimeError('INCREMENTAL_CACHE_CORRUPTED')
            save(a.run / 'CACHE_VERIFIED.json', manifest)
        driver = read(a.run / 'DRIVER_RECEIPT.json')
        if driver.get('finished_unix'):
            return
        time.sleep(3)


if __name__ == '__main__':
    main()
