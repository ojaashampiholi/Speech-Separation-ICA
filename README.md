# Speech Source Separation using ICA Algorithm

## Problem Statement:
A Jazz concert is held in a big auditorium and there are 20 different recording sound clips for the song that was played by the band. All of the 20 recordings were taken from different places in the auditorium. Each of the recordings have N time domain samples. There are K distinct sources that are contributing to the music that the band is playing. Separate the K sound sources from the 20 sound sources using ICA Algorithm.

## Proposed and Implemented Solution:

1. There are 20 sound clips, but we can hear from the sound clips that there are less sources than that. (i.e. N < 20).
2. Hence, the first step that we do is to implement PCA to reduce the dimensions. We can also look at the Eigen Values to determine the actual number of sources that are present in the 20 sound clips.
3. PCA Eigen Values show that there are 4 distinct sources that are present in 20 sound clips. Hence, when we use PCA whitening, we can set 4 as rank of the whitening matrix.
4. The next step is to implement the ICA Algorithm using the following update rules:
![plot](./ICA-Update_Rules.JPG)
