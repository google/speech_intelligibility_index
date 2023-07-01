# Copyright 2023 The speech_intelligibility_index Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python implementation of ANSI S3.5-1997  - Speech Intelligibility Index.

This code implements a model of speech intelligibility based on measurements
of the speech spectrum, and the underlying noise. It implements only the
one-third octave band SII procedure.

Original Matlab Version Copyright 2003-2005 Hannes Muesch
Translation to Matlab by Malcolm Slaney, Google Sound Understanding Team

Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Tuple, Union
import warnings

import jax.numpy as jnp
import numpy as np


# From Table 3 of the ANSI Standard
mid_band_freqs = [160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                  2000, 2500, 3159, 4000, 5000, 6300, 8000]

bandwidth_db = np.arange(15.65, 33.65, 1.0)  # Column 3
bandwidth_hz = 10**(bandwidth_db/10)


def speech_spectrum(vocal_effort: str) -> jnp.ndarray:
  """"This function returns the standard speech spectrum level from Table 3.

  Args:
    vocal_effort: A string naming one of the four types of voices:
      normal, raised, loud, shout.
    And a special type called "all" returns one big matrix for testing.

  Returns:
    An jnp.array giving the normal spectral level versus band number.
  """

  # pylint: disable=bad-whitespace  # To make it easier to read columns
  # This is table 3 from the ANSI standard
  ei = [[32.41, 33.81, 35.29, 30.77],
        [34.48, 33.92, 37.76, 36.65],
        [34.75, 38.98, 41.55, 42.5],
        [33.98, 38.57, 43.78, 46.51],
        [34.59, 39.11, 43.3,  47.4],
        [34.27, 40.15, 44.85, 49.24],
        [32.06, 38.78, 45.55, 51.21],
        [28.3,  36.37, 44.05, 51.44],
        [25.01, 33.86, 42.16, 51.31],
        [23,    31.89, 40.53, 49.63],
        [20.15, 28.58, 37.7,  47.65],
        [17.32, 25.32, 34.39, 44.32],
        [13.18, 22.35, 30.98, 40.8],
        [11.55, 20.15, 28.21, 38.13],
        [9.33,  16.78, 25.41, 34.41],
        [5.31,  11.47, 18.35, 28.24],
        [2.59,   7.67, 13.87, 23.45],
        [1.13,   5.07, 11.39, 20.72]]
  ei = jnp.array(ei)

  vocal_effort = vocal_effort.lower()
  if vocal_effort == 'normal':
    return ei[:, 0]
  elif vocal_effort == 'raised':
    return ei[:, 1]
  elif vocal_effort == 'loud':
    return ei[:, 2]
  elif vocal_effort == 'shout':
    return ei[:, 3]
  elif vocal_effort == 'all':
    return ei  # For Testing
  else:
    raise ValueError(f'Identifier string {vocal_effort} not recognized')


band_importance_names = ['standard',  # Table 3
                         'nns', 'cid-22', 'nu6',  # All the rest from Table B.2
                         'drt', 'spin', 'short', 'spin', 'cst']


def band_importance(test_number: Union[int, List[float], str,
                                       jnp.ndarray]) -> jnp.ndarray:
  """Return a weighting vector for different kinds of tests.

  Different speech materials have different average spectra, and this data
  reflects those differences.

  Args:
    test_number: Either a name from the list above, an index from the table
      below, or the actual desired band-importance as an 18-element vector.

      One of 8 different standard test for calculating importance:
      Standard
        1: Average speech as specified in the band-importance columne of Table 3
   		  2: various nonsense syllable tests where most English
  	       phonemes occur equally often (NNS from Table B.2)
        3: CID-22
        4: NU6
        5: Diagnostic Rhyme test
        6: short passages of easy reading material
        7: SPIN
      Non_standard:
        8: CST (Table 1 of Sherbecoe and Studebaker, Ear and Hearing 2003)
           Article includes one extra band below this table and one above.
  Returns:
    An np vector showing the importance of each band.
  """
  if isinstance(test_number, str):
    if test_number not in band_importance_names:
      raise ValueError(f'Test name {test_number} not recognized, should be '
                       f'one of: {band_importance_names}')
    else:
      test_number = band_importance_names.index(test_number)

  if isinstance(test_number, int):
    if test_number < 1 or test_number > 8:
      raise ValueError('Test number must be between 1 and 8')

    # pylint: disable=bad-whitespace  # To make it easier to read columns
    band_importance_array = [
        [0.0083, 0,      0.0365, 0.0168, 0,      0.0114, 0,       0.0082],
        [0.0095, 0,      0.0279, 0.013,  0.024,  0.0153, 0.0255,  0.0168],
        [0.015,  0.0153, 0.0405, 0.0211, 0.033,  0.0179, 0.0256,  0.0255],
        [0.0289, 0.0284, 0.05,   0.0344, 0.039,  0.0558, 0.036,   0.0374],
        [0.044,  0.0363, 0.053,  0.0517, 0.0571, 0.0898, 0.0362,  0.0637],
        [0.0578, 0.0422, 0.0518, 0.0737, 0.0691, 0.0944, 0.0514,  0.0694],
        [0.0653, 0.0509, 0.0514, 0.0658, 0.0781, 0.0709, 0.0616,  0.0529],
        [0.0711, 0.0584, 0.0575, 0.0644, 0.0751, 0.066,  0.077,   0.0374],
        [0.0818, 0.0667, 0.0717, 0.0664, 0.0781, 0.0628, 0.0718,  0.0441],
        [0.0844, 0.0774, 0.0873, 0.0802, 0.0811, 0.0672, 0.0718,  0.0784],
        [0.0882, 0.0893, 0.0902, 0.0987, 0.0961, 0.0747, 0.1075,  0.1035],
        [0.0898, 0.1104, 0.0938, 0.1171, 0.0901, 0.0755, 0.0921,  0.1023],
        [0.0868, 0.112,  0.0928, 0.0932, 0.0781, 0.082,  0.1026,  0.0926],
        [0.0844, 0.0981, 0.0678, 0.0783, 0.0691, 0.0808, 0.0922,  0.0738],
        [0.0771, 0.0867, 0.0498, 0.0562, 0.048,  0.0483, 0.0719,  0.0596],
        [0.0527, 0.0728, 0.0312, 0.0337, 0.033,  0.0453, 0.0461,  0.0454],
        [0.0364, 0.0551, 0.0215, 0.0177, 0.027,  0.0274, 0.0306,  0.0365],
        [0.0185, 0,      0.0253, 0.0176, 0.024,  0.0145, 0,       0.0275]
        ]
    band_importance_array = jnp.array(band_importance_array)
    return band_importance_array[:, test_number-1]
  else:
    importance = jnp.asarray(test_number)
    if importance.shape[0] != 18:
      raise ValueError('Supplied band importance must have 18 values')
  return importance


def input_5p1(ssl, nsl=None, insertion_gain=None,
              distance=None, hearing_threshold=None,
              num_channels=1) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Implementation of Section 5.1 of ANSI S3.5-1997.

  "Determination of equivalent speech, noise, and threshold spectrum levels:
  method based on the direct measurement/estimation of noise and speech
  spectrum levels at the listener's position."

  The output produced by this function is intended to be passed to the
  "sii" function.

  Args:
    ssl: speech_spectrum level (dB) Section 3.6, either:
        (a) String arguments use the "standard speech spectrum level for the
            stated vocal effort" as defined in Table 3 of the standard.
                        'normal'
                        'raised'
                        'loud'
                        'shout'
            These levels assume the reference communication condition (talker
            1 meter in front of listener).

        (b) A vector with 18 numbers stating the speech spectrum levels in dB
            SPL at the listener's position in bands 1 through 18.
            The distance, if specified, will be ignored and a warning message
            is printed.

    nsl: noise_spectrum_level (dB): Section 3.13 and Section 5.1.4
            A vector with 18 numbers stating the Noise Spectrum Levels in dB
            SPL at the listener's position in bands 1 through 18.
            The default Equivalent Noise Spectrum Level is -50 dB in all 18
            bands (see note in Section 4.2).

    insertion_gain: (dB) Section 3.28
            A vector with 18 numbers stating the Insertion Gain in dB in
            bands 1 through 18.
            The default Insertion Gain is 0 dB in all 18 bands.

    distance: distance from source to listener
            A positive scalar, stating the distance between the sound source
            and the listener's head (meters). If omitted the distance for
            the reference communication situation (1 meter) is assumed. This
            parameter is valid only when ssl is specified as one of the named
            vocal efforts, i.e., according to clause (a) in the specification
            of "SSL"; it will be ignored otherwise.

    hearing_threshold: [dB HL] Section 3.22
            A vector with 18 numbers stating the Hearing Threshold Levels in
            dBHL in bands 1 through 18.
            The default Equivalent Hearing Threshold Level is 0 dBHL in all
            18 bands.

    num_channels:
            A scalar, having a value of either 1 or 2, indicating the
            listening mode: b = 1 --> monaural listening, b = 2 --> binaural
            listening. Monaural listening is the default.

  Returns:
    A tuple of three JAX arrays giving the speech spectrum level (ssl),
    noise spectrum level (msl) and hearing_threshold.
  """
  if not ssl:
    raise ValueError('The Speech Spectrum Level, ssl, must be specified')

  if isinstance(ssl, str):
    ssl = speech_spectrum(ssl)  # Standard spectra are used
    distance = distance or 1.0  # Reference communication situation assumed
    if insertion_gain is None:
      insertion_gain = jnp.zeros(18)
    ssl = ssl - 20*jnp.log10(distance) + insertion_gain  # Eq. 16
  else:
    ssl = jnp.asarray(ssl)  # Speech Spectrum measured at listener's head
    if insertion_gain is None:
      insertion_gain = jnp.zeros(18)
    ssl = ssl + insertion_gain  # Eq. 17
    if distance:
      warnings.warn('Distance parameter is inappropriately '
                    'specified and ignored!')

  # DERIVE EQUIVALENT NOISE SPECTRUM LEVEL
  if nsl is None:
    nsl = -50*jnp.ones(18)
  else:
    nsl = jnp.asarray(nsl) + jnp.asarray(insertion_gain)  # Eq. 18

  # DERIVE EQUIVALENT HEARING THRESHOLD LEVEL
  if hearing_threshold is None:
    hearing_threshold = jnp.zeros(18)
  else:
    hearing_threshold = jnp.asarray(hearing_threshold)

  if num_channels != 1 and num_channels != 2:
    raise ValueError('Invalid value of num_channels specified!')

  if num_channels == 2:                                # Binaural listening
    hearing_threshold -= 1.7                           # Section 5.1.5

  return (ssl, nsl, hearing_threshold)


def input_5p2(csns, mtf, gain=None,
              hearing_threshold=None,
              binaural=False)-> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Implementation of Section 5.2 of ANSI S3.5-1997.

  Determination of equivalent speech, noise, and threshold spectrum levels:
  method based on MTFI/CSNSL measurements at the listener's position.  The
  output produced by this function is intended to be passed to the function
  sii.

  Copyright 2005 Hannes Muesch & Pat Zurek. Translated to Python in 2023.

  Args:
    csns: Combined Speech and Noise Spectrum Level [dB] (Section 3.17)
      A row or column vector with 18 numbers stating the Combined Speech and
      Noise Spectrum Level in dB in bands 1 through 18.
    mtf:  Modulation Transfer Function for Intensity (Section 3.31)
      An 18x9 matrix containing the Modulation Transfer Function for Intensity
      at the 18 third-octave audio frequencies and the 9 modulation frequencies
      specified in Section 5.2.3.3 (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0 and
      16.0Hz).
    gain: Insertion Gain [dB] (Section 3.28)
      A row or column vector with 18 numbers stating the Insertion Gain in dB in
      bands 1 through 18. If this identifier is omitted, a default Insertion
      Gain of 0 dB is assumed in all 18 bands.
    hearing_threshold: Hearing Threshold Level [dB HL] (Section 3.22)
       A row or column vector with 18 numbers stating the Hearing Threshold
       Levels in dBHL in bands 1 through 18. If this identifier is omitted, a
       default Equivalent Hearing Threshold Level of 0 dBHL is assumed in all
       18 bands.
    binaural: A boolean indicating the listening mode: if False then monaural
      listening, binaural listening if True. Monaural is the default.

  Returns:
    A tuple of three JAX arrays giving the speech spectrum level (ssl),
    noise spectrum level (msl) and hearing_threshold.
  """
  csns = jnp.asarray(csns)
  if csns.shape != (18,):
    raise ValueError('Combined Speech and Noise Spectrum Level: Vector size '
                     'incorrect')

  mtf = jnp.asarray(mtf)
  if mtf.shape != (18, 9):
    raise ValueError('Modulation Transfer Function for Intensity: '
                     'Matrix size incorrect')

  if gain is None:
    gain = jnp.zeros(18, dtype=float)
  else:
    gain = jnp.asarray(gain)
  if gain.shape != (18,):
    raise ValueError('Insertion Gain: Vector size incorrect')

  # DERIVE EQUIVALENT HEARING THRESHOLD LEVEL
  if hearing_threshold is None:
    hearing_threshold = jnp.zeros(18, dtype=float)
  else:
    hearing_threshold = jnp.asarray(hearing_threshold)
  if hearing_threshold.shape != (18,):
    raise ValueError('Hearing Threshold Level: Vector size incorrect')

  if binaural:  # Binaural listening
    hearing_threshold = hearing_threshold - 1.7  # Section 5.1.5

  eps = jnp.finfo(float).eps
  # apparent speech-to-noise ratio (5.2.3.5, Eq. 22)
  snr = 10*jnp.log10((mtf+eps) / (1-mtf+eps))
  # limit to range -15 .. +15 dB
  snr = jnp.minimum(15.0, jnp.maximum(-15.0, snr))
  snr = jnp.mean(snr, axis=1)  # Average across modulation frequencies (5.2.3.6)
  # Equivalent Speech and Equivalent Noise spectra (5.2.3.8, Eq. 23
  esnr = snr + 10*jnp.log10(10**(csns/10) / (1 + 10**(snr/10)) )
  nsl = esnr - snr                                             # ... and Eq 24)

  ###
  # Note: Eq. 23 and 24 represent the Equivalent Speech and Equivalent Noise
  esnr = esnr + gain
  # spectra only if the insertion gain is 0dB. Provisions for insertion gain
  # are not explicitly shown in the standard. However, they are necessary and
  # are applied here
  nsl = nsl + gain

  return esnr, nsl, hearing_threshold


def ff2ed() -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Free-field to eardrum transfer function.

  Returned at the 1/3 oct frequencies between 160 Hz and 8000Hz, inclusive.
  (From Table 3
  Returns:
    Blah
    Blah
  """

  tf = [0, 0.50, 1.00, 1.40, 1.50, 1.80, 2.40, 3.10, 2.60, 3.00, 6.10,
        12.00, 16.80, 15.00, 14.30, 10.70, 6.40, 1.80]
  fc = [160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
        2000, 2500, 3150, 4000, 5000, 6300, 8000]
  return jnp.asarray(tf), jnp.asarray(fc)


def input_5p3(csns, mtf, hearing_threshold=None,
              binaural=False)-> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Implementation of Section 5.3 of ANSI S3.5-1997.

  Determination of equivalent speech, noise, and threshold spectrum levels:
  method based on MTFI/CSNSL measurements at the eardrum of the listener.

  The output produced by this function is intended to be passed to the function
  sii.

  Args:
    csns: Combined Speech and Noise Spectrum Level [dB] (Section 3.17)
      A row or column vector with 18 numbers stating the Combined Speech and
      Noise Spectrum Level in dB in bands 1 through 18.
    mtf:  Modulation Transfer Function for Intensity (Section 3.31)
      An 18x9 matrix containing the Modulation Transfer Function for Intensity
      at the 18 third-octave audio frequencies and the 9 modulation frequencies
      specified in Section 5.2.3.3 (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0 and
      16.0Hz).
    hearing_threshold: Hearing Threshold Level [dB HL] (Section 3.22)
       A row or column vector with 18 numbers stating the Hearing Threshold
       Levels in dBHL in bands 1 through 18. If this identifier is omitted, a
       default Equivalent Hearing Threshold Level of 0 dBHL is assumed in all
       18 bands.
    binaural: A boolean indicating the listening mode: if False then monaural
      listening, binaural listening if True. Monaural is the default.

  Returns:
    A tuple of three JAX arrays giving the speech spectrum level (ssl),
    noise spectrum level (msl) and hearing_threshold.
  """
  if csns.shape != (18,):
    raise ValueError('Combined Speech and Noise Spectrum Level: Vector size '
                     'incorrect')

  mtf = jnp.asarray(mtf)
  if mtf.shape != (18, 9):
    raise ValueError('Modulation Transfer Function for Intensity: '
                     'Matrix size incorrect')

  # DERIVE EQUIVALENT HEARING THRESHOLD LEVEL
  if hearing_threshold is None:
    hearing_threshold = jnp.zeros(18, dtype=float)
  else:
    hearing_threshold = jnp.asarray(hearing_threshold)
  if hearing_threshold.shape != (18,):
    raise ValueError('Hearing Threshold Level: Vector size incorrect')

  if binaural:  # Binaural listening
    hearing_threshold = hearing_threshold - 1.7  # Section 5.1.5

  eps = jnp.finfo(float).eps
  # apparent speech-to-noise ratio (5.2.3.5, Eq. 22)
  snr = 10*jnp.log10((mtf+eps) / (1-mtf+eps))
  snr = jnp.minimum(15, jnp.maximum(-15, snr))  # limit to range -15 ... +15 dB
  snr = jnp.mean(snr, axis=1)  # Average across modulation frequencies (5.2.3.6)
  # APPARENT Speech and APPARENT Noise spectra (5.3.3.3, Eq. 25 and Eq 26)
  esnr = snr + 10*jnp.log10(10**(csns/10) / (1 + 10**(snr/10)))
  nsl = esnr - snr

  tf, _ = ff2ed()
  esnr = esnr - tf  # Equivalent Speech Spectrum Level (Eq 27)
  nsl = nsl - tf   # Equivalent Noise Spectrum Level (Eq 28)

  return esnr, nsl, hearing_threshold


def sii(ssl, nsl=None, hearing_threshold=None,
        band_importance_function: int = 1) -> float:
  """Compute the speech intelligibility index according to the ANSI standard.

  Methods for calculation of the Speech Intelligibility Index"
    (ANSI S3.5-1997). Python implementation of Section 4.

  The speech and noise spectrum levels are the power per Hz in dB.
  Spectrum levels are the amount of power (in dB) per Hz.  This can be written
  as the sum of the power in a band divided by the bandwidth, in the limit as
  the bandwidth goes to zero. More practically, the power in a band is the
  spectrum *level* multiplied by the filter's bandwidth.

  Note: The remaining sections of the standard, which provide means to
  calculate input parameters required by the "core" SII procedure of Section
  4, are implemented in seperate scripts:
    Section 5.1 in script Input_5p1.m "method based on the direct measurement/
      estimation of noise and speech spectrum levels at the listener's position"
    Section 5.2 in script Input_5p2.m "method based on MTFI/CSNSL measurements
      at the listener's position"
    Section 5.3 in script Input_5p3.m "method based on MTFI/CSNSL measurements
      at the eardrum of the listener"

  Args:
    ssl: 'E' Equivalent Speech Spectrum Level (Section 3.6 in the standard)
      A vector of 18 numbers stating the Equivalent Speech Spectrum Levels
      in dB in bands 1 through 18.

    nsl: Equivalent Noise Spectrum Level (Section 3.15 in the standard)
      A vector of 18 numbers stating the Equivalent Noise Spectrum Levels in
      dB in bands 1 through 18. If this identifier is omitted, a default
      Equivalent Noise Spectrum Level of -50 dB is assumed in all 18 bands
      (see note in Section 4.2).

    hearing_threshold: Equivalent Hearing Threshold Level [dBHL]
      (Section 3.23 in the standard)
      A vector of 18 numbers stating the Equivalent Hearing Threshold Levels
      in dBHL in bands 1 through 18. If this identifier is omitted, a default
      Equivalent Hearing Threshold Level of 0 dBHL is assumed in all 18 bands.

    band_importance_function: Section 3.1 in the standard
      A scalar having a value between 1 and 8. The
      Band-importance functions associated with each scalar are:
        1:	Average speech as specified in Table 3 (DEFAULT)
        2:	various nonsense syllable tests where most English phonemes occur
            equally often (as specified in Table B.2)
        3:	CID-22 (as specified in Table B.2)
        4:	NU6 (as specified in Table B.2)
        5:	Diagnostic Rhyme test (as specified in Table B.2)
        6:	short passages of easy reading material (as specified in Table B.2)
        7:	SPIN (as specified in Table B.2)
        Non_standard
        8: CST (Table 1 of Sherbecoe and Studebaker, Ear and Hearing 2003)

  Returns:
    The function returns the SII of the specified listening condition, which
    is a value in the interval [0, 1].

  REMINDER OF DEFINITIONS & MEANINGS:

  Equivalent Speech Spectrum Level, E-prime
    The SII calculation is based on free-field levels, even though
    the quantity relevant for perception and intelligibility
    is the level at the listener's eardrum.

    The Equivalent Speech Spectrum Level is the speech spectrum level at the
    center of the listener's head (when the listener is temporarily absent)
    that produces in an average human with unoccluded ears an eardrum speech
    level equal to the eardrum speech level actually present in the listening
    situation to be analyzed.

    Before the SII can be applied to a given listening situation, the
    corresponding Equivalent Speech Spectrum Level must be derived. For
    example, when speech is presented over insert earphones (earphones inside
    the earcanal), only the speech spectrum level at the eardrum is known.
    Using the inverse of the freefield-to-eardrum transfer function (Table 3
    of the standard) this eardrum level must be "projected" into the
    freefield, yielding the Equivalent Speech Spectrum Level.

    Sections 5.1, 5.2, and 5.3 of the standard give three examples of how to
    derive the Equivalent Speech Spectrum Level from physical measurements.
    The standard allows the use of alternative transforms, such as the one
    illustrated above, where appropriate.

  Equivalent Noise Spectrum Level, N-prime
    Similar to the Equivalent Speech Spectrum Level, the Equivalent Noise
    Spectrum Level is the noise spectrum level at the center of the listener's
    head (when the listener is temporarily absent) that produces an eardrum
    noise level equal to the eardrum noise level actually present in the
    listening situation to be analyzed. Sections 5.1, 5.2, and 5.3 give three
    examples of how to derive the Equivalent Speech Spectrum Level from
    physical measurements.
  """
  ################# VERIFY INTEGRITY OF INPUT VARIABLES ######################

  ssl = jnp.asarray(ssl)
  if nsl is None:
    nsl = -50*jnp.ones(18)
  else:
    nsl = jnp.asarray(nsl)

  if hearing_threshold is None:
    hearing_threshold = jnp.zeros(18)
  else:
    hearing_threshold = jnp.asarray(hearing_threshold)

  if nsl.shape != (18,):
    raise ValueError('Equivalent Noise Spectrum Level: Vector size incorrect')
  if hearing_threshold.shape != (18,):
    raise ValueError('Equivalent Hearing Threshold Level: '
                     'Vector size incorrect')
  if ssl.shape != (18,):
    raise ValueError('Equivalent Speech Spectrum Level: Vector size incorrect')

  ################# IMPLEMENTATION OF SPEECH INTELLIGIBILITY INDEX ############

  # THE NUMBERS IN PARENTHESIS REFER TO THE SECTIONS IN THE ANSI STANDARD
  # pylint: disable=invalid-name  # variable names are from the standard

  # Define band center frequencies for 1/3rd octave procedure (Table 3)
  f = jnp.array([160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
                 2500, 3150, 4000, 5000, 6300, 8000])

  # Define Internal Noise Spectrum Level (Table 3)
  X = jnp.array([0.6, -1.7, -3.9, -6.1, -8.2, -9.7, -10.8, -11.9, -12.5,
                 -13.5, -15.4, -17.7, -21.2, -24.2, -25.9, -23.6, -15.8, -7.1])

  # Self-Speech Masking Spectrum (4.3.2.1 Eq. 5)
  V = ssl - 24

  # 4.3.2.2
  B = jnp.maximum(V, nsl)

  # Calculate slope parameter Ci (4.3.2.3 Eq. 7)
  C = 0.6*(B + 10*jnp.log10(f) - 6.353) - 80

  # Initialize Equivalent Masking Spectrum Level (4.3.2.4)
  # Z = jnp.zeros(18)
  # Z[0] = B[0]
  Z = [B[0],]
  # Calculate Equivalent Masking Spectrum Level (4.3.2.5 Eq. 9)
  for i in range(1, 18):
    Z.append(10*jnp.log10(10**(0.1*nsl[i]) +
                          jnp.sum(10**(0.1*(B[:i] + 3.32*C[:i]*
                                            jnp.log10(0.89*f[i]/f[:i]))))))
  Z = jnp.asarray(Z)
  # Equivalent Internal Noise Spectrum Level (4.4 Eq. 10)
  X = X + hearing_threshold

  # Disturbance Spectrum Level (4.5)
  D = jnp.maximum(Z, X)

  # Level Distortion Factor (4.6 Eq. 11)
  L = 1 - (ssl - speech_spectrum('normal') - 10)/160
  L = jnp.minimum(1, L)

  # 4.7.1 Eq. 12
  K = (ssl-D+15)/30
  K = jnp.minimum(1, jnp.maximum(0, K))

  # Band Audibility Function (7.7.2 Eq. 13)
  A = L*K

  # Speech Intelligibility Index (4.8 Eq. 14)
  return jnp.sum(band_importance(band_importance_function)*A)
