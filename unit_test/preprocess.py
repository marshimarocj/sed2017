import os


'''func
'''


'''expr
'''
def generate_08_script():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  data_dir = os.path.join(root_dir, 'video')
  lst_file = os.path.join(data_dir, 'video.lst')
  video_dir = '/data/'
  out_dir = '/data/preprocess'
  img_name = 'diva_preprocess_opencv3:latest'
  out_file = 'preprocess08.sh'

  with open(lst_file) as f, open(out_file, 'w') as fout:
    for line in f:
      line = line.strip()

      out_lst_file = os.path.join(data_dir, line + '.lst')
      with open(out_lst_file, 'w') as _fout:
        _fout.write(line + '\n')
      _out_lst_file = os.path.join('/data', line + '.lst')

      cmd = [
        'docker', 'run',
        '--rm', '-it',
        '-v', data_dir + ':/data',
        '-v', '/dev/null:/dev/raw1394',
        img_name,
        '--segment_clip', '1',
        '--clip_length', '6000',
        '--video_dir', video_dir, 
        '--video_lst_file', _out_lst_file,
        '--out_dir', out_dir
      ]
      fout.write(' '.join(cmd) + '\n')


def missing_videos_in_preprocess():
  root_dir = '/usr0/home/jiac/data/sed/tst2017' # aladdin3
  lst_file = os.path.join(root_dir, 'refined.2017.lst')
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
        fout.write(name + '\n')


if __name__ == '__main__':
  # generate_dev08_script()
  missing_videos_in_preprocess()
