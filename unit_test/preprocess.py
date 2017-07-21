import os


'''func
'''


'''expr
'''
def missing_videos_in_preprocess():
  root_dir = '/usr0/home/jiac/data/sed/tst2017' # aladdin3
  lst_file = os.path.join(root_dir, '2017.refined.lst')
  preprocess_root_dir = os.path.join(root_dir, 'dev09_preprocess')
  out_file = os.path.join(root_dir, 'dev09_preprocess.short.lst')

  with open(lst_file) as f, open(out_file, 'w') as fout:
    for line in f:
      name = line.strip()
      preprocess_dir = os.path.join(preprocess_root_dir, name)
      lst_file = os.path.join(preprocess_dir, 'frame_5.lst')
      # if not os.path.exists(preprocess_dir):
      #   print name
      if not os.path.exists(lst_file):
        fout.write(name + '.mov.deint.avi\n')


if __name__ == '__main__':
  missing_videos_in_preprocess()
