# About

This repo contains code for recutting video randomly according to a
number of user-defined preferences. The primary function accepts as
series of durations (e.g. 5 frames, 100 frames, 1000 frames, 25
frames) and randomly picks contiguous selections of frames from the
original movie to construct the series of new 'scenes' of correct
duration. Selections of frames from the original is done without
overlaps (i.e. every frame from the original is used exactly once int
he output). The user can also choose to bias the random cuts to match
cuts or scene changes in the original video. The class also exposes
methods to assist the user in generating desirable profiles of new
'scene' durations.

# Dependencies

The VideoRandomizer makes use of [`ffmpeg`](https://github.com/FFmpeg/FFmpeg) for handling audio.
