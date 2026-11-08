import hashlib,json,pathlib,shutil,tarfile
root=pathlib.Path('/mnt/home/liyuqin/USERDIR/vln_baselines_20261103')
src=root/'src/Falcon'
assert shutil.disk_usage(root).free>4*2**30+2*2**30
archives=[
 ('hm3d-minival-habitat-v0.2.tar','760540ae32722c2a06a6598154aafff3c84afed6ce37a201129e8b36d9867e08',src/'data/scene_datasets/hm3d/minival'),
 ('falcon_val_hm3dminival9.tar.gz','678e3d24649cd439dac74574a70d6eca0357bccfae9df9e40b1b4876b37195aa',src),
 ('falcon_robot_assets_minimal.tar.gz','1a06dc5bc2937f0f639a0b062952afc50d138fab00102aef56f19cfa6cdbc3c5',src),
 ('navthinker_source.tar.gz','ecf785b8b83b17343aadb26fee70ed788621220473f96c89d35c0d096f8ed116',root/'src/NavThinker'),
]
records=[]
for name,expected,target in archives:
 p=root/'inputs'/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
 target.mkdir(parents=True,exist_ok=True)
 with tarfile.open(p) as tf:
  for m in tf.getmembers():
   rel=pathlib.Path(m.name)
   assert not rel.is_absolute() and '..' not in rel.parts and (m.isfile() or m.isdir()),m.name
  tf.extractall(target)
 records.append({'archive':str(p),'sha256':expected,'target':str(target)})
with tarfile.open(root/'inputs/input_manifests.tar.gz') as tf:
 for m in tf.getmembers():
  assert m.isfile() and len(pathlib.Path(m.name).parts)==1 and m.name.endswith('.json')
 tf.extractall(root/'inputs/manifests')
nav=root/'src/NavThinker'
nm=json.loads((root/'inputs/manifests/navthinker_source_manifest.json').read_text())
for row in nm['files']:
 p=nav/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
manifest=json.loads((root/'inputs/manifests/falcon_val_hm3dminival9_manifest.json').read_text())
val=src/'data/scene_datasets/hm3d/val';val.mkdir(exist_ok=True)
for row in manifest['scenes']:
 scene=pathlib.Path(row['scene_id']).parent.name
 link=val/scene
 target=src/'data/scene_datasets/hm3d/minival'/scene
 assert target.is_dir()
 if not link.exists():link.symlink_to(target,target_is_directory=True)
 assert link.resolve()==target.resolve()
mesh_records=[]
for row in manifest['scenes']:
 rel=row['scene_id'].removeprefix('data/scene_datasets/')
 mesh=src/'data/scene_datasets'/rel
 for p in [mesh,mesh.with_suffix('.navmesh')]:
  assert p.is_file()
  mesh_records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
result={'status':'DATA_PREPARED_NO_MODEL_EXECUTION','archives':records,'scene_count':manifest['scene_count'],
        'episode_count':manifest['episode_count'],'nav_source_files_verified':len(nm['files']),'mesh_files':mesh_records}
(root/'prep_logs/data_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
