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


if __name__ == '__main__':
  pass
