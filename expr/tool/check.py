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


if __name__ == '__main__':
  missing_videos_in_preprocess()
