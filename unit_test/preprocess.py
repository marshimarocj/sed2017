import os
import json


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


def lnk_short_video_imgs_to_ease_tar():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  lst_file = os.path.join(root_dir, 'tst2017', 'dev09_preprocess.short.lst')
  preprocess_dir = os.path.join(root_dir, 'video', 'dev09', 'preprocess')
  out_root_dir = os.path.join(root_dir, 'tst2017', 'short_video_img_5')

  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find('.')
      videoname = line[:pos]
      out_dir = os.path.join(out_root_dir, videoname)
      if not os.path.exists(out_dir):
        os.mkdir(out_dir)
      src_img_dir = os.path.join(preprocess_dir, videoname, 'frame_5')
      dst_img_dir = os.path.join(out_dir, 'frame_5')
      os.symlink(src_img_dir, dst_img_dir)
      src_img_lst_file = os.path.join(preprocess_dir, videoname, 'frame_5.lst')
      dst_img_lst_file = os.path.join(out_dir, 'frame_5.lst')
      os.symlink(src_img_lst_file, dst_img_lst_file)
      src_img_lst_file = os.path.join(preprocess_dir, videoname, 'frame_25.lst')
      dst_img_lst_file = os.path.join(out_dir, 'frame_25.lst')
      os.symlink(src_img_lst_file, dst_img_lst_file)


def lnk_long_video_imgs_to_ease_tar():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin3
  lst_file = os.path.join(root_dir, 'tst2017', '2017.refined.lst')
  preprocess_dir = os.path.join(root_dir, 'tst2017', 'dev09_preprocess')
  out_root_dir = os.path.join(root_dir, 'tst2017', 'long_video_img_5')

  with open(lst_file) as f:
    for line in f:
      videoname = line.strip()

      src_img_lst_file = os.path.join(preprocess_dir, videoname, 'frame_5.lst')
      if not os.path.exists(src_img_lst_file):
        continue
      out_dir = os.path.join(out_root_dir, videoname)
      if not os.path.exists(out_dir):
        os.mkdir(out_dir)
      dst_img_lst_file = os.path.join(out_dir, 'frame_5.lst')
      os.symlink(src_img_lst_file, dst_img_lst_file)

      src_img_dir = os.path.join(preprocess_dir, videoname, 'frame_5')
      dst_img_dir = os.path.join(out_dir, 'frame_5')
      os.symlink(src_img_dir, dst_img_dir)


def get_numframe_file():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  lst_file = os.path.join(root_dir, 'tst2017', '2017.refined.lst')
  preprocess_dir = os.path.join(root_dir, 'video', 'dev09', 'preprocess')

  with open(lst_file) as f:
    for line in f:
      videoname = line.strip()
      meta_file = os.path.join(preprocess_dir, videoname, 'meta.json')
      with open(meta_file) as f:
        data = json.load(f)
      total_frame = data['total_frames']
      out_file = os.path.join(preprocess_dir, '%s.num_frame'%videoname)
      with open(out_file, 'w') as fout:
        fout.write('%d\n'%total_frame)


def lnk_to_solve_000001bug():
  root_dir = '/home/jiac/data2/sed' # gpu9
  lst_file = os.path.join(root_dir, '2017.refined.lst')
  img_root_dir = os.path.join(root_dir, 'image_per_5', 'tst2017')

  with open(lst_file) as f:
    for line in f:
      videoname = line.strip()
      src_img_file = os.path.join(img_root_dir, videoname, 'frame_5', '000001.jpg')
      dst_img_file = os.path.join(img_root_dir, videoname, 'frame_5', '000000.jpg')
      os.symlink(src_img_file, dst_img_file)


if __name__ == '__main__':
  # missing_videos_in_preprocess()
  # lnk_short_video_imgs_to_ease_tar()
  # lnk_long_video_imgs_to_ease_tar()
  # get_numframe_file()
  lnk_to_solve_000001bug()
