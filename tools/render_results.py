"""Render the README chart from the exported evaluation records (standard library)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
metrics = json.loads((ROOT / 'evidence/subset174/aggregate_metrics.json').read_text())
coverage = json.loads((ROOT / 'evidence/accepted471/COVERAGE.json').read_text())
svg = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="490" viewBox="0 0 1040 490" role="img" aria-labelledby="title desc">
<title id="title">Falcon evaluation results and coverage</title>
<desc id="desc">174 episodes in 9 scenes: success 55.75%, SPL 50.89%, personal space compliance 89.44%, human collision 40.80%. Separately, 471 of 1087 episodes are validated in the full evaluation.</desc>
<rect width="1040" height="490" rx="18" fill="#f5f7fa"/>
<g font-family="Palatino, Georgia, serif" fill="#193044">
<text x="38" y="48" font-size="26" font-weight="bold">Falcon · Checkpoint Evaluation</text>
<text x="38" y="78" font-size="15" fill="#586c7c">174 episodes / 9 scenes · seed 100 · 4 environments · non-random subset</text>
''']
series = [('SR', 'success', 'Task success', '#277969', 'higher'),
          ('SPL', 'spl', 'Path efficiency', '#3f7f9e', 'higher'),
          ('PSC', 'psc', 'Personal space', '#677da1', 'higher'),
          ('H-Coll', 'human_collision', 'Collision episodes', '#bc7650', 'lower')]
for i, (name, key, caption, color, direction) in enumerate(series):
    x = 38 + i * 245
    value = metrics[key] * 100
    svg.append(f'''<rect x="{x}" y="104" width="229" height="208" rx="12" fill="white"/>
<text x="{x+18}" y="137" font-size="19" font-weight="bold">{name}</text>
<text x="{x+18}" y="187" font-size="36" fill="{color}">{value:.2f}%</text>
<rect x="{x+18}" y="211" width="193" height="8" rx="4" fill="#e9edf2"/>
<rect x="{x+18}" y="211" width="{193*value/100:.4f}" height="8" rx="4" fill="{color}"/>
<text x="{x+18}" y="252" font-size="15">{caption}</text>
<text x="{x+18}" y="279" font-size="13" fill="#6c7b87">{direction.capitalize()} is better · 0–100%</text>''')
n = coverage['episodes']
total = coverage['expected_full_episodes']
svg.append(f'''<text x="38" y="358" font-size="19" font-weight="bold">Full evaluation coverage</text>
<text x="1000" y="358" text-anchor="end" font-size="18">{n:,} / {total:,} episodes</text>
<rect x="38" y="379" width="964" height="14" rx="7" fill="#dde5ed"/>
<rect x="38" y="379" width="{964*n/total:.4f}" height="14" rx="7" fill="#277969"/>
<text x="38" y="424" font-size="15" fill="#586c7c">24 / 56 scenes validated · remaining 616 episodes · paused</text>
<text x="38" y="459" font-size="13" fill="#6c7b87">Subset scores and full-evaluation coverage are separate runs with overlapping episodes. Snapshot: 2026-11-08.</text>
</g></svg>''')
output = ROOT / 'assets/falcon-results.svg'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('\n'.join(svg), encoding='utf-8')
print(output)
