import os
import xml.etree.ElementTree
import tarfile
import sys
sys.path.append('../')

import api.db


'''func
'''
event2lid = {
  'CellToEar': 1,
  'Embrace': 2,
  'Pointing': 3,
  'PersonRuns': 4 
}


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


def generate_csv():
  root_dir = '/home/jiac/data/sed2017' # rocks
  predict_dir = os.path.join(root_dir, 'expr', 'twostream', 'eev08_full')
  video_dir = os.path.join(root_dir, 'video')
  lst_file = os.path.join(root_dir, 'eev08-1.lst')
  track_dir = os.path.join(root_dir, 'tracking')

  threshold = 0.5

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      if 'CAM4' not in name:
        names.append(name)

  events = {}
  for event in event2lid:
    lid = event2lid[event]
    events[lid] = event

  for name in names:
    predict_file = os.path.join(predict_dir, name + '.npz')
    data = np.load(predict_file)
    ids = data['ids']
    predicts = data['predicts']

    track_db_file = os.path.join(track_dir, name + '.25.forward.backward.square.npz')
    track_db = api.db.TrackDb()
    track_db.load(track_db_file)

    start_frame2score = {}
    num = ids.shape[0]
    for i in range(num):
      id = ids[i]
      predict = predicts[i]
      lid = np.argmax(predict)
      score = np.max(predict)
      if lid > 0:
        track = track_db.trackid2track[id]
        start_frame = track.start_frame
        if start_frame not in start_frame2score:
          start_frame2score[start_frame] = {'score': score, 'lid': lid}
        if score > start_frame2score[start_frame]['score']:
          start_frame2score[start_frame] = {'score': score, 'lid': lid}

    num_frame_file = os.path.join(video_dir, name + '.num_frame')
    with open(num_frame_file) as f:
      line = f.readline()
      line = line.strip()
      num_frame = int(line)

    out_file = os.path.join(predict_dir, name + '.csv')
    with open(out_file, 'w') as fout:
      fout.write('"ID","EventType","Framespan","DetectionScore","DetectionDecision"\n')
      cnt = 1
      for start_frame in start_frame2score:
        lid = start_frame2score[start_frame]['lid']
        event = events[lid]
        score = start_frame2score[start_frame]['score']
        end_frame = min(start_frame + 25, num_frame)
        decision = score >= threshold
        fout.write('"%d","%s","%d:%d","%f","%d"\n'%(cnt, event, start_frame, end_frame, score, decision))

        cnt += 1


if __name__ == '__main__':
  # extract_tst_videos()
  # tar_tst_videos()
  # lnk_2017_tst_flow_ft_for_transfer()
  generate_csv()
