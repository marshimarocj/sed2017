import os
import shutil


'''func
'''


'''expr
'''
def generate_script():
  root_dir = '/usr0/home/jiac/data/sed/tst2017' # aladdin3
  # video_lst_file = os.path.join(root_dir, '2017.refined.lst')
  video_lst_file = os.path.join(root_dir, 'dev09_preprocess.long.lst')
  preprocess_dir = os.path.join(root_dir, 'dev09_preprocess')
  script_dir = os.path.join(root_dir, 'script')
  # root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  # video_lst_file = os.path.join(root_dir, 'tst2017', 'dev09_preprocess.short.lst')
  # preprocess_dir = os.path.join(root_dir, 'video', 'dev09', 'preprocess')
  # script_dir = os.path.join(root_dir, 'tst2017', 'script')

  image_name = 'diva_tracking:v1'

  with open(video_lst_file) as f:
    for line in f:
      videoname = line.strip()
      pos = videoname.find('.')
      videoname = videoname[:pos]
      clip_lst_file = os.path.join(preprocess_dir, videoname, 'clip_6000_100.lst')
      out_file = os.path.join(script_dir, videoname + '.sh')
      if not os.path.exists(clip_lst_file):
        # print videoname
        continue
      print videoname, out_file
      out_dir = os.path.join(root_dir, 'tst2017', 'tracking', videoname)
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
            # os.path.join('/data/video/dev09/preprocess', videoname, 'clip_6000_100', clip),
            # os.path.join('/data/tst2017/person_detect_0.8', videoname),
            # os.path.join('/data/tst2017/person_detect_0.8', videoname, '%s.frame_25.lst'%videoname),
            # os.path.join('/data/tst2017/tracking', videoname),
            '--track_duration', '100',
          ]
          fout.write(' '.join(cmd) + '\n')


def group_script():
  # root_dir = '/usr0/home/jiac/data/sed/tst2017' # aladdin3
  # video_lst_file = os.path.join(root_dir, '2017.refined.lst')
  # script_dir = os.path.join(root_dir, 'script')
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  video_lst_file = os.path.join(root_dir, 'tst2017', 'dev09_preprocess.short.lst')
  script_dir = os.path.join(root_dir, 'tst2017', 'script')

  cnt = 0
  cmds = []
  idx = 0
  with open(video_lst_file) as flst:
    for line in flst:
      videoname = line.strip()
      pos = videoname.find('.')
      videoname = videoname[:pos]
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


def generate_25fps_lst_from_5fps_lst():
  # root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  # video_lst_file = os.path.join(root_dir, 'tst2017', 'dev09_preprocess.short.lst')
  # preprocess_root_dir = os.path.join(root_dir, 'video', 'dev09', 'preprocess')
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  video_lst_file = os.path.join(root_dir, 'tst2017', '2017.refined.lst')
  preprocess_root_dir = os.path.join(root_dir, 'dev09_preprocess')

  with open(video_lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find('.')
      name = line[:pos]
      src_lst_file = os.path.join(preprocess_root_dir, name, 'frame_5.lst')
      # dst_lst_file = os.path.join(preprocess_root_dir, name, 'frame_25.lst')
      dst_lst_file = os.path.join(preprocess_root_dir, name, 'frame_100.lst')
      with open(src_lst_file) as f, open(dst_lst_file, 'w') as fout:
        cnt = 0
        for line in f:
          # if cnt % 5 == 0:
          if cnt % 20 == 0:
            fout.write(line)
          cnt += 1


def normalize_bbox_name():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  lst_file = os.path.join(root_dir, 'tst', 'person_detect.lst')
  bbox_dir = os.path.join(root_dir, 'tst', 'person_detect')
  out_lst_file = os.path.join(root_dir, 'tst', 'person_detect.norm.lst')

  with open(lst_file) as f, open(out_lst_file, 'w') as fout:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      frame = int(name)
      fout.write('%06d.txt\n'%frame)

      src_file = os.path.join(bbox_dir, '%d.txt'%frame)
      dst_file = os.path.join(bbox_dir, '%06d.txt'%frame)
      # os.symlink(src_file, dst_file)
      shutil.copyfile(src_file, dst_file)


def normalize_track_name():
  # root_dir = '/home/jiac/sdb/jiac/data/sed' # gpu1
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_file = os.path.join(root_dir, 'tst2017', '2017.refined.lst')
  track_root_dir = os.path.join(root_dir, 'tst2017', 'tracking')

  with open(lst_file) as f:
    for line in f:
      videoname = line.strip()

      track_dir = os.path.join(track_root_dir, videoname)
      names = os.listdir(track_dir)
      out_lst_file = os.path.join(track_root_dir, videoname + '.lst')
      with open(out_lst_file, 'w') as fout:
        for name in names:
          name, _ = os.path.splitext(name)
          if 'M' in name:
            continue

          frame = int(name)
          src_file = os.path.join(track_dir, '%d.npy'%frame)
          dst_file = os.path.join(track_dir, '%06d.npy'%frame)
          if os.path.exists(dst_file):
            os.remove(dst_file)
          # os.symlink(src_file, dst_file)
          shutil.copyfile(src_file, dst_file)
          fout.write('%06d.npy\n'%frame)


if __name__ == '__main__':
  # generate_script()
  # group_script()
  generate_25fps_lst_from_5fps_lst()
  # normalize_bbox_name()
  # normalize_track_name()
