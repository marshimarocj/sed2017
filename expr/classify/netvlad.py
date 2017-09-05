import os
import cPickle


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
def generate_label2lid_file():
  root_dir = '/home/jiac/data/sed' # xiaojun
  out_file = os.path.join(root_dir, 'meta', 'label2lid.pkl')
  with open(out_file, 'w') as fout:
    cPickle.dump(event2lid, fout)


if __name__ == "__main__":
  generate_label2lid_file()
