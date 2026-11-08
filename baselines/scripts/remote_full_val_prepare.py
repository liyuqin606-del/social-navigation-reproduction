import hashlib,json,pathlib,shutil,tarfile
r=pathlib.Path('/mnt/home/liyuqin/USERDIR/vln_baselines_20261103')
assert shutil.disk_usage(r).free>4*2**30+3*2**30
metadata=r/'inputs/full_val_manifests';metadata.mkdir(exist_ok=True)
with tarfile.open(r/'inputs/full_val_inputs_final.tar.gz') as t:
 for m in t.getmembers():assert m.isfile() and len(pathlib.Path(m.name).parts)==1
 t.extractall(metadata)
geometry=json.loads((metadata/'falcon_full_val_geometry_selected56_20261103_manifest.json').read_text())
p=r/'inputs/falcon_full_val_geometry_selected56_20261103.tar.gz'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
assert sha(p)=='1258c805efa713a40ec709dcf7ebc2c9f1cbe416d1f40985f655bf3ba17acb38'
target=r/'fullval_assets/scene_datasets/hm3d/val';target.mkdir(parents=True,exist_ok=True)
expected={row['path']:row for row in geometry['files']}
with tarfile.open(p) as t:
 members=t.getmembers();assert {m.name for m in members}==set(expected)
 for m in members:assert m.isfile() and not pathlib.Path(m.name).is_absolute() and '..' not in pathlib.Path(m.name).parts
 t.extractall(target)
records=[]
for name,row in expected.items():
 f=target/name;assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],name
 records.append({'path':str(f),'bytes':row['bytes'],'sha256':row['sha256']})
episodes=metadata/'falcon_full_val_20261103.tar.gz'
assert sha(episodes)=='68cb3008b95f3503ae7bea57358e8fb2ce175ad4f794893e53e5389e1a27f70f'
with tarfile.open(episodes) as t:
 for m in t.getmembers():assert m.isfile() and not pathlib.Path(m.name).is_absolute() and '..' not in pathlib.Path(m.name).parts
 t.extractall(r/'src/Falcon')
out={'status':'FULL_VAL_ASSETS_VERIFIED_NO_MODEL_EXECUTION','scene_count':56,'episode_count':1087,
     'scenes_dir':str(r/'fullval_assets/scene_datasets'),'data_path':str(r/'src/Falcon/data/datasets/pointnav/social-hm3d-fullval/val/val.json.gz'),
     'mesh_files':records,'geometry_archive_sha256':sha(p),'episode_archive_sha256':sha(episodes)}
(r/'prep_logs/full_val_preparation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
