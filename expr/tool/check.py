import os


'''func
'''


'''expr
'''
def missing_videos_in_preprocess():
  root_dir = '/home/jiac/data/sed/video' # xiaojun
  lst_file = os.path.join(root_dir, 'video2017.lst')
  preprocess_root_dir = os.path.join(root_dir, 'dev09_preprocess')

  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find('.')
      name = line[:pos]
      preprocess_dir = os.path.join(preprocess_root_dir, name)
      if not os.path.exists(preprocess_dir):
        print name


def mkdir_for_c3d_sync():
  root_dir = '/home/jiac/data/sed2017' # rocks
  lst_files = [
    os.path.join(root_dir, 'dev08-1.lst'),
    os.path.join(root_dir, 'eev08-1.lst'),
  ]
  for lst_file in lst_files:
    with open(lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        if 'CAM4' not in name:
          out_dir = os.path.join(root_dir, 'c3d', name)
          os.mkdir(out_dir)


if __name__ == '__main__':
  # missing_videos_in_preprocess()
  mkdir_for_c3d_sync()
