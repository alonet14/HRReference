#!python
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3

# This script contains classes that define all the parameters for
# a radar system

# This script requires that 'numpy' be installed within the Python
# environment you are running this script in.

# This file can be imported as a module and contains the following
# class:

# * Transmitter - A class defines parameters of a radar transmitter
# * Receiver - A class defines parameters of a radar receiver
# * Radar - A class defines basic parameters of a radar system

# ----------
# RadarSimPy - A Radar Simulator Built with Python
# Copyright (C) 2018 - 2021  Zhengyu Peng
# E-mail: zpeng.me@gmail.com
# Website: https://zpeng.me

# `                      `
# -:.                  -#:
# -//:.              -###:
# -////:.          -#####:
# -/:.://:.      -###++##:
# ..   `://:-  -###+. :##:
#        `:/+####+.   :##:
# .::::::::/+###.     :##:
# .////-----+##:    `:###:
#  `-//:.   :##:  `:###/.
#    `-//:. :##:`:###/.
#      `-//:+######/.
#        `-/+####/.
#          `+##+.
#           :##:
#           :##:
#           :##:
#           :##:
#           :##:
#            .+:


import numpy as np
import scipy.constants as const
from scipy.interpolate import interp1d

from .util import cal_phase_noise


class Transmitter:
    """
    A class defines basic parameters of a radar transmitter

    :param f:
        Waveform frequency (Hz).
        If ``f`` is a single number, all the pulses have
        the same center frequency.

        For linear modulation, specify ``f`` with ``[f_start, f_stop]``.

        ``f`` can alse be a 1-D array of an arbitrary waveform, specify
        the time with ``t``.
    :type f: float or numpy.1darray
    :param t:
        Timing of each pulse (s).
    :type t: float or numpy.1darray
    :param numpy.1darray f_offset:
        Frequency offset for each pulse (Hz). The length must be the same
        as ``pulses``.
    :param float tx_power:
        Transmitter power (dBm)
    :param float prp:
        Pulse repetition period (s). ``prp >=
        pulse_length``. If it is ``None``, ``prp =
        pulse_length``.

        ``prp`` can alse be a 1-D array to specify
        different repetition period for each pulse. In this case, the
        length of the 1-D array should equals to the length
        of ``pulses``
    :type repetitions_period: float or numpy.1darray
    :param int pulses:
        Total number of pulses
    :param numpy.1darray pn_f:
        Frequency of the phase noise (Hz)
    :param numpy.1darray pn_power:
        Power of the phase noise (dB/Hz)
    :param list[dict] channels:
        Properties of transmitter channels

        [{

        - **location** (*numpy.1darray*) --
            3D location of the channel [x. y. z] (m)
        - **delay** (*float*) --
            Transmit delay (s). ``default 0``
        - **azimuth_angle** (*numpy.1darray*) --
            Angles for azimuth pattern (deg). ``default [-90, 90]``
        - **azimuth_pattern** (*numpy.1darray*) --
            Azimuth pattern (dB). ``default [0, 0]``
        - **elevation_angle** (*numpy.1darray*) --
            Angles for elevation pattern (deg). ``default [-90, 90]``
        - **elevation_pattern** (*numpy.1darray*) --
            Elevation pattern (dB). ``default [0, 0]``
        - **pulse_amp** (*numpy.1darray*) --
            Relative amplitude sequence for pulse's amplitude modulation.
            The array length should be the same as `pulses`. ``default 0``
        - **pulse_phs** (*numpy.1darray*) --
            Phase code sequence for pulse's phase modulation (deg).
            The array length should be the same as `pulses`. ``default 0``
        - **t_mod** (*numpy.1darray*) --
            Time stamps for waveform modulation (s). ``default None``
        - **phs** (*numpy.1darray*) --
            Phase scheme for waveform modulation (deg). ``default None``
        - **amp** (*numpy.1darray*) --
            Relative amplitude scheme for waveform modulation. ``default None``

        }]

    :ivar numpy.1darray fc:
        Center frequency array for the pulses
    :ivar float pulse_length:
        Dwell time of each pulse (s)
    :ivar float bandwidth:
        Bandwith of each pulse (Hz)
    :ivar float tx_power:
        Transmitter power (dBm)
    :ivar numpy.1darray prp:
        Pulse repetition period (s)
    :ivar int pulses:
        Total number of pulses
    :ivar list[dict] channels:
        Properties of transmitter channels
    :ivar int channel_size:
        Number of transmitter channels
    :ivar numpy.2darray locations:
        3D location of the channels. Size of the aray is
        ``[channel_size, 3 <x, y, z>]`` (m)
    :ivar list[numpy.1darray] az_angles:
        Angles for each channel's azimuth pattern (deg)
    :ivar list[numpy.1darray] az_patterns:
        Azimuth pattern for each channel (dB)
    :ivar list[numpy.1darray] el_angles:
        Angles for each channel's elevation pattern (deg)
    :ivar list[numpy.1darray] el_patterns:
        Elevation pattern for each channel (dB)
    :ivar list az_func:
        Azimuth patterns' interpolation functions
    :ivar list el_func:
        Elevation patterns' interpolation functions
    :ivar numpy.1darray antenna_gains:
        Antenna gain for each channel (dB).
        Antenna gain is ``max(az_pattern)``
    :ivar list[numpy.1darray] pulse_phs:
        Phase code sequence for phase modulation (deg)
    :ivar numpy.1darray chip_length:
        Length for each phase code (s)
    :ivar numpy.1darray delay:
        Delay for each channel (s)
    :ivar numpy.1darray polarization:
        Antenna polarization ``[x, y, z]``.

        - Horizontal polarization: ``[1, 0, 0]``
        - Vertical polarization: ``[0, 0, 1]``

    :ivar float wavelength:
        Wavelength (m)
    :ivar float slope:
        Waveform slope, ``bandwidth / pulse_length``

    **Waveform**

    ::

        |                       prp
        |                  +-----------+
        |
        |            +---f[1]--->  /            /            /
        |                         /            /            /
        |                        /            /            /
        |                       /            /            /
        |                      /            /            /     ...
        |                     /            /            /
        |                    /            /            /
        |                   /            /            /
        |      +---f[0]--->/            /            /
        |
        |                  +-------+
        |                 t[0]    t[1]
        |
        |    Pulse         +--------------------------------------+
        |    modulation    |pulse_amp[0]|pulse_amp[1]|pulse_amp[2]|  ...
        |                  |pulse_phs[0]|pulse_phs[1]|pulse_phs[2]|  ...
        |                  +--------------------------------------+
        |
        |    Waveform      +--------------------------------------+
        |    modulation    |           amp / phs / t_mod          |  ...
        |                  +--------------------------------------+

    Tips:

    - Set ``bandwidth`` to 0 get a tone waveform for a Doppler radar

    """

    def __init__(self,
                 f,
                 t,
                 f_offset=None,
                 tx_power=0,
                 prp=None,
                 pulses=1,
                 pn_f=None,
                 pn_power=None,
                 channels=[dict(location=(0, 0, 0))]):

        self.tx_power = tx_power
        self.pulses = pulses
        self.channels = channels

        if isinstance(f, (list, tuple, np.ndarray)):
            self.f = np.array(f)
        else:
            self.f = np.array([f, f])

        if isinstance(t, (list, tuple, np.ndarray)):
            self.t = np.array(t)
            self.t = self.t - \
                self.t[0]
        else:
            self.t = np.array([0, t])

        if len(self.f) != len(self.t):
            raise ValueError(
                'Length of `f`, and `t` should be the same')

        if f_offset is not None:
            if isinstance(f_offset, (list, tuple, np.ndarray)):
                self.f_offset = np.array(f_offset)
            else:
                self.f_offset = f_offset+np.zeros(pulses)
        else:
            self.f_offset = np.zeros(pulses)

        self.bandwidth = np.max(self.f) - np.min(self.f)
        self.pulse_length = self.t[-1]-self.t[0]

        self.fc_0 = (np.min(self.f)+np.max(self.f))/2
        self.fc_vect = (np.min(self.f)+np.max(self.f))/2+self.f_offset
        self.fc_frame = (np.min(self.fc_vect)+np.max(self.fc_vect))/2

        self.pn_f = pn_f
        self.pn_power = pn_power

        # Extend `prp` to a numpy.1darray.
        # Length equels to `pulses`
        if prp is None:
            self.prp = self.pulse_length + np.zeros(pulses)
        else:
            if isinstance(prp, (list, tuple, np.ndarray)):
                if len(prp) != pulses:
                    raise ValueError(
                        'Length of `prp` should equal to the \
                            length of `pulses`.')
                else:
                    self.prp = prp
            else:
                self.prp = prp + np.zeros(pulses)

        if np.min(self.prp < self.pulse_length):
            raise ValueError(
                '`prp` should be larger than `pulse_length`.')

        self.chirp_start_time = np.cumsum(
            self.prp)-self.prp[0]

        self.max_code_length = 0

        self.channel_size = len(self.channels)
        self.locations = np.zeros((self.channel_size, 3))

        self.mod = []
        self.pulse_mod = np.ones(
            (self.channel_size, self.pulses), dtype=complex)
        self.antenna = []

        self.az_patterns = []
        self.az_angles = []
        self.el_patterns = []
        self.el_angles = []
        self.az_func = []
        self.el_func = []
        self.pulse_phs = []
        self.chip_length = []
        self.polarization = np.zeros((self.channel_size, 3))
        self.antenna_gains = np.zeros((self.channel_size))
        self.grid = []
        self.delay = np.zeros(self.channel_size)
        for tx_idx, tx_element in enumerate(self.channels):
            self.delay[tx_idx] = self.channels[tx_idx].get('delay', 0)

            mod_enabled = True
            amp = self.channels[tx_idx].get('amp', None)
            if amp is not None:
                if isinstance(amp, (list, tuple, np.ndarray)):
                    amp = np.array(amp)
                else:
                    amp = np.array([amp, amp])
            else:
                mod_enabled = False

            phs = self.channels[tx_idx].get('phs', None)
            if phs is not None:
                if isinstance(phs, (list, tuple, np.ndarray)):
                    phs = np.array(phs)
                else:
                    phs = np.array([phs, phs])
            else:
                mod_enabled = False

            if phs is not None and amp is None:
                amp = np.ones_like(phs)
                mod_enabled = True
            elif phs is None and amp is not None:
                phs = np.zeros_like(amp)
                mod_enabled = True

            t_mod = self.channels[tx_idx].get('t_mod', None)
            if t_mod is not None:
                if isinstance(t_mod, (list, tuple, np.ndarray)):
                    t_mod = np.array(t_mod)
                else:
                    t_mod = np.array([0, t_mod])
            else:
                mod_enabled = False

            if mod_enabled:
                mod_var = amp*np.exp(1j*phs/180*np.pi)
            else:
                mod_var = None

            self.mod.append({
                'enabled': mod_enabled,
                'var': mod_var,
                't': t_mod
            })

            self.pulse_mod[tx_idx, :] = self.channels[tx_idx].get(
                'pulse_amp', np.ones((self.pulses))) * \
                np.exp(1j * self.channels[tx_idx].get(
                    'pulse_phs', np.zeros((self.pulses))) / 180 * np.pi)

            self.locations[tx_idx, :] = np.array(
                tx_element.get('location'))
            self.polarization[tx_idx, :] = np.array(
                tx_element.get('polarization', np.array([0, 0, 1])))
            self.az_angles.append(
                np.array(self.channels[tx_idx].get('azimuth_angle',
                                                   np.arange(-90, 91, 1))))
            self.az_patterns.append(
                np.array(self.channels[tx_idx].get('azimuth_pattern',
                                                   np.zeros(181))))

            self.antenna_gains[tx_idx] = np.max(self.az_patterns[-1])

            self.az_patterns[-1] = self.az_patterns[-1] - \
                np.max(self.az_patterns[-1])
            self.az_func.append(
                interp1d(self.az_angles[-1], self.az_patterns[-1],
                         kind='linear', bounds_error=False, fill_value=-10000)
            )
            self.el_angles.append(
                np.array(self.channels[tx_idx].get('elevation_angle',
                                                   np.arange(-90, 91, 1))))
            self.el_patterns.append(
                np.array(self.channels[tx_idx].get('elevation_pattern',
                                                   np.zeros(181))))
            self.el_patterns[-1] = self.el_patterns[-1] - \
                np.max(self.el_patterns[-1])
            self.el_func.append(
                interp1d(
                    self.el_angles[-1],
                    self.el_patterns[-1]-np.max(self.el_patterns[-1]),
                    kind='linear', bounds_error=False, fill_value=-10000)
            )

            self.grid.append(self.channels[tx_idx].get('grid', 1))

        self.box_min = np.min(self.locations, axis=0)
        self.box_max = np.max(self.locations, axis=0)


class Receiver:
    """
    A class defines basic parameters of a radar receiver

    :param float fs:
        Sampling rate (sps)
    :param float noise_figure:
        Noise figure (dB)
    :param float rf_gain:
        Total RF gain (dB)
    :param float load_resistor:
        Load resistor to convert power to voltage (Ohm)
    :param float baseband_gain:
        Total baseband gain (dB)
    :param list[dict] channels:
        Properties of transmitter channels

        [{

        - **location** (*numpy.1darray*) --
            3D location of the channel [x. y. z] (m)
        - **azimuth_angle** (*numpy.1darray*) --
            Angles for azimuth pattern (deg). ``default [-90, 90]``
        - **azimuth_pattern** (*numpy.1darray*) --
            Azimuth pattern (dB). ``default [0, 0]``
        - **elevation_angle** (*numpy.1darray*) --
            Angles for elevation pattern (deg). ``default [-90, 90]``
        - **elevation_pattern** (*numpy.1darray*) --
            Elevation pattern (dB). ``default [0, 0]``

        }]

    :ivar float fs:
        Sampling rate (sps)
    :ivar float noise_figure:
        Noise figure (dB)
    :ivar float rf_gain:
        Total RF gain (dB)
    :ivar float load_resistor:
        Load resistor to convert power to voltage (Ohm)
    :ivar float baseband_gain:
        Total baseband gain (dB)
    :ivar float noise_bandwidth:
        Bandwidth in calculating the noise (Hz).
        ``noise_bandwidth = fs / 2``
    :ivar list[dict] channels:
        Properties of receiver channels
    :ivar int channel_size:
        Total number of receiver channels
    :ivar numpy.2darray locations:
        3D location of the channels. Size of the aray is
        ``[channel_size, 3 <x, y, z>]`` (m)
    :ivar list[numpy.1darray] az_angles:
        Angles for each channel's azimuth pattern (deg)
    :ivar list[numpy.1darray] az_patterns:
        Azimuth pattern for each channel (dB)
    :ivar list[numpy.1darray] el_angles:
        Angles for each channel's elevation pattern (deg)
    :ivar list[numpy.1darray] el_patterns:
        Elevation pattern for each channel (dB)
    :ivar list az_func:
        Azimuth patterns' interpolation functions
    :ivar list el_func:
        Elevation patterns' interpolation functions
    :ivar numpy.1darray antenna_gains:
        Antenna gain for each channel (dB).
        Antenna gain is ``max(az_pattern)``

    **Receiver noise**

    ::

        |           + n1 = 10*log10(Boltzmann_constant * Ts * 1000)
        |           |      + 10*log10(noise_bandwidth)  (dBm)
        |           v
        |    +------+------+
        |    |rf_gain      |
        |    +------+------+
        |           | n2 = n1 + noise_figure + rf_gain (dBm)
        |           v n3 = 1e-3 * 10^(n2/10) (Watts)
        |    +------+------+
        |    |mixer        |
        |    +------+------+
        |           | n4 = sqrt(n3 * load_resistor) (V)
        |           v
        |    +------+------+
        |    |baseband_gain|
        |    +------+------+
        |           | noise amplitude (peak to peak)
        |           v n5 = n4 * 10^(baseband_gain / 20) * sqrt(2) (V)

    """

    def __init__(self, fs,
                 noise_figure=10,
                 rf_gain=0,
                 load_resistor=500,
                 baseband_gain=0,
                 channels=[dict(location=(0, 0, 0))]):
        self.fs = fs
        self.noise_figure = noise_figure
        self.rf_gain = rf_gain
        self.load_resistor = load_resistor
        self.baseband_gain = baseband_gain
        self.noise_bandwidth = self.fs / 2

        # additional receiver parameters

        self.channels = channels
        self.channel_size = len(self.channels)
        self.locations = np.zeros((self.channel_size, 3))
        self.az_patterns = []
        self.az_angles = []
        self.az_func = []
        self.el_patterns = []
        self.el_angles = []
        self.antenna_gains = np.zeros((self.channel_size))
        self.el_func = []
        for rx_idx, rx_element in enumerate(self.channels):
            self.locations[rx_idx, :] = np.array(
                rx_element.get('location'))
            self.az_angles.append(
                np.array(self.channels[rx_idx].get('azimuth_angle',
                                                   np.arange(-90, 91, 1))))
            self.az_patterns.append(
                np.array(self.channels[rx_idx].get('azimuth_pattern',
                                                   np.zeros(181))))
            self.antenna_gains[rx_idx] = np.max(self.az_patterns[-1])
            self.az_patterns[-1] = self.az_patterns[-1] - \
                np.max(self.az_patterns[-1])
            self.az_func.append(
                interp1d(self.az_angles[-1], self.az_patterns[-1],
                         kind='linear', bounds_error=False, fill_value=-10000)
            )
            self.el_angles.append(
                np.array(self.channels[rx_idx].get('elevation_angle',
                                                   np.arange(-90, 91, 1))))
            self.el_patterns.append(
                np.array(self.channels[rx_idx].get('elevation_pattern',
                                                   np.zeros(181))))
            self.el_patterns[-1] = self.el_patterns[-1] - \
                np.max(self.el_patterns[-1])
            self.el_func.append(
                interp1d(
                    self.el_angles[-1],
                    self.el_patterns[-1]-np.max(self.el_patterns[-1]),
                    kind='linear', bounds_error=False, fill_value=-10000)
            )

        self.box_min = np.min(self.locations, axis=0)
        self.box_max = np.max(self.locations, axis=0)


class Radar:
    """
    A class defines basic parameters of a radar system

    :param Transmitter transmitter:
        Radar transmiter
    :param Receiver receiver:
        Radar Receiver
    :param time:
        Radar firing time instances / frames
    :type time: float or numpy.1darray
    :param int seed:
        Seed for noise generator

    :ivar Transmitter transmitter:
        Radar transmiter
    :ivar Receiver receiver:
        Radar Receiver
    :ivar int samples_per_pulse:
        Number of samples in one pulse
    :ivar int channel_size:
        Total number of channels.
        ``channel_size = transmitter.channel_size * receiver.channel_size``
    :ivar numpy.2darray virtual_array:
        Locations of virtual array elements. [channel_size, 3 <x, y, z>]
    :ivar float max_range:
        Maximum range for an FMCW mode (m).
        ``max_range = c * fs * pulse_length / bandwidth / 2``
    :ivar float unambiguous_speed:
        Unambiguous speed (m/s).
        ``unambiguous_speed = c / prp / fc / 2``
    :ivar float range_resolution:
        Range resolution (m).
        ``range_resolution = c / 2 / bandwidth``
    :ivar numpy.3darray timestamp:
        Timestamp for each samples. Frame start time is
        defined in ``time``.
        ``[channes/frames, pulses, samples]``

        *Channel/frame order in timestamp*

        *[0]* ``Frame[0] -- Tx[0] -- Rx[0]``

        *[1]* ``Frame[0] -- Tx[0] -- Rx[1]``

        ...

        *[N]* ``Frame[0] -- Tx[1] -- Rx[0]``

        *[N+1]* ``Frame[0] -- Tx[1] -- Rx[1]``

        ...

        *[M]* ``Frame[1] -- Tx[0] -- Rx[0]``

        *[M+1]* ``Frame[1] -- Tx[0] -- Rx[1]``

    """

    def __init__(self,
                 transmitter,
                 receiver,
                 time=0,
                 seed=None,
                 **kwargs):

        self.validation = kwargs.get('validation', False)

        self.transmitter = transmitter
        self.receiver = receiver

        self.samples_per_pulse = int(self.transmitter.pulse_length *
                                     self.receiver.fs)

        self.t_offset = np.array(time)
        self.frames = np.size(time)

        if self.transmitter.bandwidth > 0:
            self.max_range = (const.c * self.receiver.fs *
                              self.transmitter.pulse_length /
                              self.transmitter.bandwidth / 2)
            self.unambiguous_speed = const.c / \
                self.transmitter.prp[0] / \
                self.transmitter.fc_0 / 2
            self.range_resolution = const.c / 2 / self.transmitter.bandwidth
        else:
            self.max_range = 0
            self.unambiguous_speed = 0
            self.range_resolution = 0

        # virtual array
        self.channel_size = self.transmitter.channel_size * \
            self.receiver.channel_size
        self.virtual_array = np.repeat(
            self.transmitter.locations, self.receiver.channel_size,
            axis=0) + np.tile(self.receiver.locations,
                              (self.transmitter.channel_size, 1))

        self.box_min = np.min(
            [self.transmitter.box_min, self.receiver.box_min], axis=0)
        self.box_max = np.max(
            [self.transmitter.box_min, self.receiver.box_max], axis=0)

        self.timestamp = self.gen_timestamp()
        self.pulse_phs = self.cal_frame_phases()
        # self.code_timestamp = self.cal_code_timestamp()
        self.noise = self.cal_noise()

        if len(self.transmitter.f) > 2:
            fun_f_t = interp1d(self.transmitter.t,
                               self.transmitter.f, kind='linear')
            self.t = np.linspace(
                self.transmitter.t[0],
                self.transmitter.t[-1],
                self.samples_per_pulse*100)
            self.f = fun_f_t(self.t)

        else:
            self.f = self.transmitter.f
            self.t = self.transmitter.t

        self.delta_f = np.ediff1d(self.f, to_begin=0)
        self.delta_t = np.ediff1d(self.t, to_begin=0)
        self.k = self.delta_f[1:]/self.delta_t[1:]

        # if hasattr(self.transmitter.fc, '__len__'):
        self.fc_mat = np.tile(
            self.transmitter.fc_vect[np.newaxis, :, np.newaxis],
            (self.channel_size, 1, self.samples_per_pulse)
        )

        self.f_offset_mat = np.tile(
            self.transmitter.f_offset[np.newaxis, :, np.newaxis],
            (self.channel_size, 1, self.samples_per_pulse)
        )

        beat_time_samples = np.arange(0,
                                      self.samples_per_pulse,
                                      1) / self.receiver.fs
        self.beat_time = np.tile(
            beat_time_samples[np.newaxis, np.newaxis, ...],
            (self.channel_size, self.transmitter.pulses, 1)
        )

        if self.transmitter.pn_f is not None and \
                self.transmitter.pn_power is not None:
            dummy_sig = np.ones(
                (self.channel_size*self.frames*self.transmitter.pulses,
                 self.samples_per_pulse))
            self.phase_noise = cal_phase_noise(
                dummy_sig,
                self.receiver.fs,
                self.transmitter.pn_f,
                self.transmitter.pn_power,
                seed=seed,
                validation=self.validation)
            self.phase_noise = np.reshape(self.phase_noise, (
                self.channel_size*self.frames,
                self.transmitter.pulses,
                self.samples_per_pulse
            ))
        else:
            self.phase_noise = None

    def gen_timestamp(self):
        """
        Generate timestamp

        :return:
            Timestamp for each samples. Frame start time is
            defined in ``time``.
            ``[channes/frames, pulses, samples]``
        :rtype: numpy.3darray
        """

        channel_size = self.channel_size
        rx_channel_size = self.receiver.channel_size
        pulses = self.transmitter.pulses
        samples = self.samples_per_pulse
        crp = self.transmitter.prp
        delay = self.transmitter.delay
        fs = self.receiver.fs

        chirp_delay = np.tile(
            np.expand_dims(
                np.expand_dims(np.cumsum(crp)-crp[0], axis=1),
                axis=0),
            (channel_size, 1, samples))

        tx_idx = np.arange(0, channel_size)/rx_channel_size
        tx_delay = np.tile(
            np.expand_dims(
                np.expand_dims(delay[tx_idx.astype(int)], axis=1),
                axis=2),
            (1, pulses, samples))

        timestamp = tx_delay+chirp_delay+np.tile(
            np.expand_dims(
                np.expand_dims(np.arange(0, samples), axis=0),
                axis=0),
            (channel_size, pulses, 1))/fs

        if self.frames > 1:
            toffset = np.repeat(
                np.tile(
                    np.expand_dims(
                        np.expand_dims(self.t_offset, axis=1), axis=2), (
                        1, self.transmitter.pulses, self.samples_per_pulse
                    )), self.channel_size, axis=0)

            timestamp = np.tile(timestamp, (self.frames, 1, 1)) + toffset
        elif self.frames == 1:
            timestamp = timestamp + self.t_offset

        return timestamp

    def cal_frame_phases(self):
        """
        Calculate phase sequence for frame level modulation

        :return:
            Phase sequence. ``[channes/frames, pulses, samples]``
        :rtype: numpy.2darray
        """

        pulse_phs = self.transmitter.pulse_mod
        pulse_phs = np.repeat(pulse_phs, self.receiver.channel_size, axis=0)
        pulse_phs = np.repeat(pulse_phs, self.frames, axis=0)
        return pulse_phs

    def cal_code_timestamp(self):
        """
        Calculate phase code timing for pulse level modulation

        :return:
            Timing at the start position of each phase code.
            ``[channes/frames, max_code_length]``
        :rtype: numpy.2darray
        """

        chip_length = np.expand_dims(
            np.array(self.transmitter.chip_length),
            axis=1)
        code_sequence = chip_length*np.tile(
            np.expand_dims(
                np.arange(0, self.transmitter.max_code_length),
                axis=0),
            (self.transmitter.channel_size, 1))

        code_timestamp = np.repeat(
            code_sequence, self.receiver.channel_size, axis=0)

        code_timestamp = np.repeat(
            code_timestamp, self.frames, axis=0)

        return code_timestamp

    def cal_noise(self):
        """
        Calculate noise amplitudes

        :return:
            Peak to peak amplitude of noise.
            ``[channes/frames, pulses, samples]``
        :rtype: numpy.3darray
        """

        noise_amp = np.zeros([
            self.channel_size,
            self.transmitter.pulses,
            self.samples_per_pulse,
        ])

        Boltzmann_const = 1.38064852e-23
        Ts = 290
        input_noise_dbm = 10 * np.log10(Boltzmann_const * Ts * 1000)  # dBm/Hz
        receiver_noise_dbm = (input_noise_dbm + self.receiver.rf_gain +
                              self.receiver.noise_figure +
                              10 * np.log10(self.receiver.noise_bandwidth) +
                              self.receiver.baseband_gain)  # dBm/Hz
        receiver_noise_watts = 1e-3 * 10**(receiver_noise_dbm / 10
                                           )  # Watts/sqrt(hz)
        noise_amplitude_mixer = np.sqrt(receiver_noise_watts *
                                        self.receiver.load_resistor)
        noise_amplitude_peak = np.sqrt(2) * noise_amplitude_mixer + noise_amp
        return noise_amplitude_peak
