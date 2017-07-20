import os


'''func
'''


'''expr
'''
def generate_script():
  root_dir = '/usr0/home/jiac/data/sed/tst2017' # aladdin3
  video_lst_file = os.path.join(root_dir, '2017.refined.lst')
  preprocess_dir = os.path.join(root_dir, 'dev09_preprocess')
  out_dir = os.path.join(root_dir, 'script')

  image_name = 'diva_tracking:v1'

  with open(video_lst_file) as f:
    for line in f:
      videoname = line.strip()
      clip_lst_file = os.path.join(preprocess_dir, videoname, 'clip_6000_100.lst')
      out_file = os.path.join(out_dir, videoname + '.sh')
      if not os.path.exists(clip_lst_file):
        # print videoname
        continue
      print videoname
      out_dir = os.path.join(root_dir, 'tracking', videoname)
      if not os.path.exists(out_dir):
        os.mkdir(out_dir)
      with open(clip_lst_file) as fclip, open(out_file, 'w') as fout:
        for line in fclip:
          clip = line.strip()
          cmd = [
            'docker', 'run', '--rm', '-it', 
            '-v', ':'.join([root_dir, '/data']),
            image_name,
            '/opt/conda/bin/python', '/workspace/diva_tracking/tracking.py',
            os.path.join('/data/dev09_preprocess', videoname, 'clip_6000_100', clip),
            os.path.join('/data/person_detect_0.8', videoname),
            os.path.join('/data/person_detect_0.8', videoname, '%s.frame_25.lst'%videoname),
            os.path.join('/data/tracking', videoname),
            '--track_duration', '25',
          ]
          fout.write(' '.join(cmd) + '\n')


def group_script():
  root_dir = '/usr0/home/jiac/data/sed/tst2017' # aladdin3
  video_lst_file = os.path.join(root_dir, '2017.refined.lst')
  script_dir = os.path.join(root_dir, 'script')

  cnt = 0
  cmds = []
  idx = 0
  with open(video_lst_file) as flst:
    for line in flst:
      videoname = line.strip()
      script_file = os.path.join(script_dir, videoname + '.sh')
      if not os.path.exists(script_file):
        continue
      with open(script_file) as f:
        for line in f:
          line = line.strip()
          cmds.append(line)

      cnt += 1
      if cnt % 6 == 0:
        out_file = '%d.sh'%idx
        with open(out_file, 'w') as fout:
          for cmd in cmds:
            fout.write(cmd + '\n')
        idx += 1
        cmds = []


if __name__ == '__main__':
  generate_script()
  # group_script()
