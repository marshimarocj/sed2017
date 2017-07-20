import os


'''func
'''


'''expr
'''
def generate_2017_script():
  root_dir = '/home/jiac/data/sed' # xiaojun
  lst_file = os.path.join(root_dir, 'video', '2017.refined.lst')
  data_root_dir = os.path.join(root_dir, 'video')
  out_file = 'person_detect.sh'

  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find('.')


def threshold_person_detect_result():
  root_dir = '/home/jiac/sdb/jiac/data/sed/tst2017_images_per_25' # gpu1
  lst_file = os.path.join(root_dir, '2017.refined.lst')
  out_root_dir = os.path.join(root_dir, 'person_detect_0.8')

  threshold = 0.8

  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find('.')
      name = line[:pos]
      img_lst_file = os.path.join(root_dir, name, 'frame_25.lst')
      person_detect_dir = os.path.join(root_dir, name, 'person_detect')
      out_dir = os.path.join(out_root_dir, name)
      if not os.path.exists(out_dir):
        os.mkdir(out_dir)
      out_lst_file = os.path.join(out_dir, name + '.frame_25.lst')
      with open(img_lst_file) as fimg, open(out_lst_file, 'w') as flst_out:
        for line in f:
          line = line.strip()
          name, _ = os.path.splitext(line)
          src_file = os.path.join(person_detect_dir, line + '.txt')
          dst_file = os.path.join(out_dir, name + '.txt')
          flst_out.write(name + '.txt\n')
          with open(src_file) as f, open(dst_file, 'w') as fout:
            for line in f:
              line = line.strip()
              data = line.split(' ')
              conf = float(data[-1])
              if conf >= threshold:
                fout.write(line + '\n')


if __name__ == '__main__':
  threshold_person_detect_result()
