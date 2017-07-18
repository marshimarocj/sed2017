import os


'''func
'''


'''expr
'''
def generate_dev09_script():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  data_dir = os.path.join(root_dir, video)
  lst_file = '/data/dev09.lst'
  video_dir = '/data/dev09'
  out_dir = '/data/dev09/preprocess'
  img_name = 'diva_preprocess_opencv3:latest'

  cmd = [
    'docker', 'run',
    '--rm', '-it',
    '-v', data_dir + ':/data',
    '-v', '/dev/null:/dev/raw1394',
    img_name,
    '--segment_clip', '1',
    '--clip_length', '6000',
    '--video_dir', video_dir, 
    '--video_lst_file', lst_file,
    '--out_dir', out_dir
  ]

def generate_dev08_script():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  data_dir = os.path.join(root_dir, video)
  lst_file = '/data/video.lst'
  video_dir = '/data/'
  out_dir = '/data/preprocess'
  img_name = 'diva_preprocess_opencv3:latest'
  out_file = 'preprocess.sh'

  with open(lst_file) as f, open(out_file, 'w') as fout:
    for line in f:
      line = line.strip()

      out_lst_file = os.path.join(root_dir, line + '.lst')
      with open(out_lst_file, 'w') as _fout:
        _fout.write(line + '\n')

      cmd = [
        'docker', 'run',
        '--rm', '-it',
        '-v', data_dir + ':/data',
        '-v', '/dev/null:/dev/raw1394',
        img_name,
        '--segment_clip', '1',
        '--clip_length', '6000',
        '--video_dir', video_dir, 
        '--video_lst_file', out_lst_file,
        '--out_dir', out_dir
      ]
      fout.write(' '.join(cmd) + '\n')


if __name__ == '__main__':
  generate_dev08_script()
