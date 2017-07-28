import os
import xml.etree.ElementTree
import tarfile


'''func
'''


'''expr
'''
def extract_tst_videos():
  root_dir = '/home/chenjia/hdd/data/sed' # earth
  lst_file = os.path.join(root_dir, '2017', 'eval07', 'expt_2017_allED_EVAL17_ENG_s-camera_NIST_1.xml')
  out_file = os.path.join(root_dir, '2017', 'eval07', 'video2017.lst')

  root = xml.etree.ElementTree.parse(lst_file).getroot()
  with open(out_file, 'w') as fout:
    for file in root.iter('{http://www.itl.nist.gov/iad/mig/tv08ecf#}filename'):
      fout.write(file.text + '\n')


def tar_tst_videos():
  root_dir = '/usr0/home/jiac/data/sed' # aladdin1
  video_dir = os.path.join(root_dir, 'video', 'dev09')
  lst_file = os.path.join(root_dir, 'video', 'dev09', 'video2017.lst')
  out_file = os.path.join(root_dir, 'video', 'dev09_2017.tar.gz')

  with open(lst_file) as f, tarfile.open(out_file, 'w:gz') as fout:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      video_file = os.path.join(video_dir, name + '.avi')
      if not os.path.exists(video_file):
        continue
      fout.add(video_file, name + '.avi')


def lnk_2017_tst_flow_ft_for_transfer():
  src_ft_dir = '/data/MM23/junweil/sed/test/feat_anet_flow_6frame'
  dst_ft_dir = '/home/jiac/data/sed2017/tst2017/feat_anet_flow_6frame'
  lst_file = '/home/jiac/data/sed2017/2017.refined.lst'

  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name = line
      src_file = os.path.join(src_ft_dir, name + '.mov.deint_1.npz')
      dst_file = os.path.join(dst_ft_dir, name + '.mov.deint_1.npz')
      os.symlink(src_file, dst_file)


if __name__ == '__main__':
  # extract_tst_videos()
  # tar_tst_videos()
  lnk_2017_tst_flow_ft_for_transfer()
