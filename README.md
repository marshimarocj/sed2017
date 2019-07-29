# SED2017
code of CMU entrance at Surveillance Event Detection TRECVID2017

depends on my personal [tensorflow experiment framework](https://github.com/marshimarocj/tf_expr_framework) and [python toolkit](https://github.com/marshimarocj/toolkit)

## api
#### db.py
* TrackDb: file system based database storing person tracking data
* ClipDb: file system based database storing video clip data
* FtDb: file system based database storing feature data
#### generator.py
* crop_instant_ft_in_track: crop feature at each time unit from the person trajectory
* crop_duration_ft_in_track: crop the feature from the person trajectory within the given time interval
* crop_clip_in_track: crop the trajectory from the video clip

## model
* netvlad: NetVLAD pooling
* attention: self-attention on temporal sequence

## expr
#### classify
* attention.py: self-attention on temporal sequence
* netvlad.py: NetVLAD pooling
* svm: vanilla SVM classifier
#### feature
* c3d_lj.py: C3D feature extraction
* paf_merge.py: Part Affinity Field of person pose feature
* twostream.py: RGB stream and flow stream feature
* vlad.py: VLAD pooling of features
