from moviepy.editor import VideoFileClip, concatenate_videoclips
from scipy.optimize import newton
from operator import itemgetter
from itertools import groupby
import numpy as np
from tqdm import tqdm


class VideoRandomizer:
    """Interface object for randomly recutting video based on user settings.

    Parameters
    ----------
    fname: String absolute path to original video object
    nFrames: Integer number of frames in original video
    nCuts: Integer number of cuts to be used in output video
    cutBias: Float 0 to 1 biasing cuts in output towards cuts in original
    sceneBias: Float 0 to 1 biasing cuts in output towards scenes in original

    Examples
    --------
    vid1 = videoRandomizer('./path/to/video1.avi')
    vid2 = videoRandomizer('./path/to/video2.mp4', nFrames=1024, nCuts=10)
    """

    def __init__(self, fname, **kwargs):
        """Initialize self. See help(type(self)) for accurate signature."""
        self.fname = fname
        self.nFrames = 0
        self.nCuts = 0
        self.cutBias = 0
        self.sceneBias = 0
        self.profile = []
        self.vidOut = []

        if 'nFrames' in kwargs:
            self.nFrames = kwargs['nFrames']
        else:
            self.set_nFrames()
        if 'nCuts' in kwargs:
            self.nCuts = kwargs['nCuts']
        if 'cutBias' in kwargs:
            self.cutBias = kwargs['cutBias']
        if 'sceneBias' in kwargs:
            self.sceneBias = kwargs['sceneBias']
        if 'profile' in kwargs:
            if kwargs['profile'][-1] != self.nFrames:
                print('Profile expected to end at {0:d}(i.e. nFrames). '
                      + 'Profile discarded. Please reset using set_profile.'
                      .format(nFrames))
            if len(kwargs['profile'] != self.nCuts + 1):
                print('Profile length should be equal to {0:d}(i.e. nCuts + 1).'
                      + 'Updating nCuts to conform to profile length.'
                      .format(self.nCuts + 1))
                self.nCuts = len(kwargs['profile']) - 1
            self.profile = kwargs['profile']

    def set_nFrames(self, *args):
        """Set nFrames from integer argument or compute from video."""
        if args:
            if len(args) > 1:
                raise TypeError(
                    'Expected 0 or 1 argument. Got ' + str(len(args)))
            else:
                self.nFrames = args[0]
        else:
            vid = VideoFileClip(self.fname)
            self.nFrames = int(vid.duration * vid.fps)

        return self

    def set_nCuts(self, nCuts):
        """Set nCuts."""
        self.nCuts = nCuts

        return self

    @staticmethod
    def _compute_profile(shape, nFrames, nCuts):
        """Compute profile of cuts."""
        if shape == 'exp':
            # calculate base in two stages
            # approximate using smooth function
            base = newton(lambda x: 1 + nFrames *
                          np.log(x) - x**nCuts, nFrames**(1 / nCuts),
                          fprime=lambda x: nFrames / x -
                          nCuts * x ** (nCuts - 1),
                          maxiter=1000)
            # solve exact discrete function using approximation
            base = newton(lambda x: np.cumsum(
                np.power(x, range(1, nCuts + 1)).astype(int))[-1] - nFrames,
                base, maxiter=1000)
            profile = np.cumsum(
                np.power(base, range(1, nCuts + 1)).astype(int))
            profile[-1] = nFrames  # force, just in case
        else:
            raise ValueError('Invalid value for shape.')

        return profile

    def set_profile(self, *args, **kwargs):
        """Set profile of cuts."""
        if len(args) == 0:
            if 'shape' in kwargs:
                print('Computing {0:s} profile for {1:d} frames and {2:d} cuts'
                      .format(kwargs['shape'], self.nFrames, self.nCuts))
                self.profile = self._compute_profile(
                    kwargs['shape'], self.nFrames, self.nCuts)
            else:
                raise ValueError(
                    'If profile is not explicit, must specify profile shape.')
        elif len(args) == 1:
            profile = args[0]
            if profile[-1] != self.nFrames:
                raise ValueError(
                    'Expected profile[-1] == {0:d} (i.e. nFrames)'
                    .format(self.nFrames))
            if len(profile) != self.nCuts + 1:
                raise ValueError(
                    'Length of profile must be {0:d} (i.e. nCuts + 1)'
                    .format(self.nCuts + 1))
            if not all(profile[k] <= profile[k + 1] for k in
                       range(len(profile) - 1)):
                raise ValueError('Profile must be sorted in ascending order.')
            if not all(isinstance(item, int) for item in profile):
                raise ValueError('Profile must contain only integers.')
            self.profile = profile
        else:
            raise TypeError(
                'Expected explicit profile or parameters for computed profile.')

        return self

    def set_cutBias(self, cutBias):
        """Set cutBias."""
        self.cutBias = cutBias

        return self

    def set_sceneBias(self, sceneBias):
        """Set sceneBias."""
        self.sceneBias = sceneBias

        return self

    def _compute_bias(self):
        return self

    @staticmethod
    def _make_space(vidOut, available, scene):
        bigGap = [0, []]
        for k, g in groupby(enumerate(available),
                            lambda item: item[0] - item[1]):
            group = list(map(itemgetter(1), g))
            if len(group) > bigGap[0]:
                bigGap = [len(group), group]
        # find scenes in vidOut containing adjacent frames
        adjFrames = (bigGap[1][0] - 1, bigGap[1][-1] + 1)
        lowBound = ([], [])
        upBound = ([], [])
        for k, g in groupby(enumerate(vidOut),
                            lambda item: item[0] - item[1]):
            indices, group = zip(*list(g))
            if adjFrames[0] in group:
                lowBound = (indices, group)
            elif adjFrames[1] in group:
                upBound = (indices, group)
        if lowBound:
            lowSpace = min([lowBound[1][0] - frame for frame
                            in filter(lambda x: x < lowBound[1][0], vidOut)])
            if lowSpace > scene - bigGap[0]:
                vidOut[list(lowBound[0])] -= scene - bigGap[0]
            else:
                vidOut[list(lowBound[0])] -= lowSpace
        if upBound:
            upSpace = min([frame - upBound[1][-1] for frame
                           in filter(lambda x: x > upBound[1][-1], vidOut)])
            vidOut[list(upBound[0])] += max(scene -
                                            bigGap[0] - lowSpace, 0)

        return vidOut

    def compute_cuts(self, **kwargs):
        """Computes location of cuts to make in original video.

        Uses profile to compute a pseudo-random recut of video of
        length nFrames. Profile specifies where the cuts will occur in
        the output. The output is a list of integers corresponding
        with frames of the original video. Cuts are discontinuities in
        the sequence.

        Cuts are computed by first taking the difference of the
        profile, a list of the number of frames between cuts. One can
        think of this list as a list of "scene" durations (hereafter
        "scenes" or "subintervals"). The task is to construct a place
        the scenes/subintervals in a (nearly) random way such that
        they cover the entire interval of the original video
        (i.e. range(nFrames)). This is accomplished by beginning with
        the largest subinterval and randomly selecting a frame, k, and
        appending range(k, k + len(subinterval)) to the output. This
        is repeated for the next largest subinterval, and so on. A
        list is created at each step of candidate frames where the
        current subinterval can be placed without overlapping previous
        selections. If this list is empty, previous selections are
        incrementally moved to create space for the current
        subinterval to fit.

        An option is included to visualize this process.
        """

        if 'vis' in kwargs and kwargs['vis']:
            pass

        self.vidOut = []
        scenes = sorted(np.diff(self.profile), reverse=True)

        if 'pbar' in kwargs and kwargs['pbar']:
            pbar = tqdm("Computing", total=len(scenes))

        frames = set(range(self.nFrames))
        used = set(self.vidOut)
        for scene in scenes:
            available = set([frame for frame in frames
                             if frame not in used])
            candidates = [frame for frame in available
                          if all(x in available for x
                                 in set(range(frame, frame + scene)))]
            if candidates:
                # assuming monotonic growth for scenes
                # prepending gives correct ordering
                # NOPE SHOULD PUSH/POP, USE STACK
                # but should really keep track of sorting and unsort later
                start = np.random.choice(candidates)
                self.vidOut[:0] = list(range(start, start + scene))
                used.update(list(range(start, start + scene)))
            else:
                # current scene doesn't fit anywhere
                # nudge existing selections out of the way
                while not candidates:
                    # _make_space errors out
                    self.vidOut = self._make_space(
                        np.array(self.vidOut), available, scene).tolist()
                    available = [frame for frame in range(self.nFrames)
                                 if frame not in self.vidOut]
                    candidates = [frame for frame in available
                                  if all(x in available for x
                                         in range(frame, frame + scene))]
                start = np.random.choice(candidates)
                self.vidOut[:0] = list(range(start, start + scene))
                used.update(list(range(start, start + scene)))
            if 'pbar' in kwargs and kwargs['pbar']:
                pbar.update(1)
            if 'vis' in kwargs and kwargs['vis']:
                pass

        return self

    @staticmethod
    def _convert_vidOut(vidOut, dt):
        # need to decide whether vidOut should be array or list
        # make operations consistent with data type
        return list(np.array(vidOut) * dt)

    def write_video(self, outname, **kwargs):
        if self.vidOut:
            if 'pbar' in kwargs and kwargs['pbar']:
                pbar = tqdm("Computing", total=len(self.vidOut))
            inVid = VideoFileClip(self.fname)
            dt = 1.0 / inVid.fps
            clips = []
            vidOut = self._convert_vidOut(self.vidOut, dt)
            for frame in vidOut:
                clips.append(inVid.subclip(frame, frame + dt))
                if 'pbar' in kwargs and kwargs['pbar']:
                    pbar.update(1)
            outVid = concatenate_videoclips(clips)
            outVid.write_videofile(outname)
        else:
            raise ValueError(
                'Output video has not been computed. Run compute_cuts.')
        return self
