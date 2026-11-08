import ast
import difflib
import hashlib
import json
from pathlib import Path

SOURCE = 'habitat-baselines/habitat_baselines/rl/ppo/falcon_evaluator.py'
SHA = 'd8bde7a4ed64917571f9125862e9446220a6576d3467e48159f78cfea47375bd'


def derive(repo, output):
    original = (Path(repo)/SOURCE).read_text()
    if hashlib.sha256(original.encode()).hexdigest() != SHA:
        raise ValueError('upstream evaluator mismatch')
    text = original
    edits = [
        ('            with inference_mode():\n',
         '            probe.before(current_episodes_info, batch, prev_actions, not_done_masks, test_recurrent_hidden_states)\n            with inference_mode():\n'),
        ('                if action_data.should_inserts is None:\n',
         '                probe.commit(action_data, action_space_lens)\n                if action_data.should_inserts is None:\n'),
        ('            outputs = envs.step(step_data)\n',
         '            outputs = envs.step(step_data)\n            probe.after(step_data, outputs, prev_actions)\n'),
        ('os.path.join("output/", "result.json")','os.path.join(os.environ["FALCON_PROBE_OUTPUT"], "result.json")'),
        ('os.path.join("output/", "actions.json")','os.path.join(os.environ["FALCON_PROBE_OUTPUT"], "actions.json")')]
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError('source anchor mismatch '+before)
        text = text.replace(before,after,1)
    marker = '        success_cal = 0 ## my added\n'
    header, body = text.split(marker,1)
    text = (header+'        from bridge import Bridge\n        probe = Bridge(agent)\n        try:\n'+
            ''.join('    '+line if line.strip() else line for line in (marker+body).splitlines(True))+
            '\n        finally:\n            probe.close()\n            envs.close()\n')
    ast.parse(text)
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'probe_evaluator.py').write_text(text)
    (output/'PATCH.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True))))
    (output/'SOURCE.json').write_text(json.dumps(dict(upstream_sha256=SHA,derived_sha256=hashlib.sha256(text.encode()).hexdigest(),upstream_modified=False),indent=2)+'\n')
