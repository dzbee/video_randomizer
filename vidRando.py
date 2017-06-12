import cv2


class videoRandomizer:
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

        if 'nFrames' in kwargs:
            self.nFrames = kwargs['nFrames']
        if 'nCuts' in kwargs:
            self.nCuts = kwargs['nCuts']
        if 'cutBias' in kwargs:
            self.cutBias = kwargs['cutBias']
        if 'sceneBias' in kwargs:
            self.sceneBias = kwargs['sceneBias']

        return self

    def set_nFrames(self, *args):
        """Set nFrames from integer argument or compute from video."""
        if args:
            if len(args) > 1:
                raise TypeError(
                    'Expected 0 or 1 argument. Got ' + str(len(args)))
            else:
                self.nFrames = args[0]
        else:
            vid = cv2.VideoCapture(self.fname)
            self.nFrames = vid.get(CV_CAP_PROP_FRAME_COUNT)

        return self

    def set_nCuts(self, nCuts):
        """Set nCuts."""
        self.nCuts = nCuts

        return self

    def set_cutBias(self, cutBias):
        """Set cutBias."""
        self.cutBias = cutBias

        return self

    def set_sceneBias(self, sceneBias):
        """Set sceneBias."""
        self.sceneBias = sceneBias

        return self

    def set_profile(self, function, **kwargs):
        return self

    def _compute_bias(self):
        return self

    def compute_cuts(self):
        return self

    def write_video(self):
        return self
