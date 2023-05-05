function [E,N,T] = Input_5p2(varargin);

% Implementation of Section 5.2 of ANSI S3.5-1997
%
% Determination of equivalent speech, noise, and threshold spectrum levels: method
% based on MTFI/CSNSL measurements at the listener's position.
%
% The output produced by this function is intended to be passed to the script "sii.m".
%
%
% Parameters are passed to the procedure through pairs of "identifier" and corresponding "argument"
% Identifiesrs are always strings. Possible identifiers are:
%
%     'P' Combined Speech and Noise Spectrum Level [dB] (Section 3.17)
%     'M' Modulation Transfer Function for Intensity (Section 3.31)
%     'G' Insertion Gain [dB] (Section 3.28)
%     'T' Hearing Threshold Level [dB HL] (Section 3.22)
%     'b' Monaural Listening? b = 1 --> monaural, b = 2 --> Binaural
%
%
% Parameters 'P' and 'M' must be specified, all others are optional. If an optional identifier is not specified a default value will be used.
% Paires of identifier and argument can occur in any order. However, if an identifier is listed, it must be followed immediately by its argument.
%
%   Arguments for 'P':
%           A row or column vector with 18 numbers stating the Combined Speech and Noise Spectrum Level in dB in bands 1 through 18.
%
%   Arguments for 'M':
%           An 18x9 matrix containing the Modulation Transfer Function for Intensity at the 18 third-octave audio frequencies and the
%           9 modulation frequencies specified in Section 5.2.3.3.
%
%   Arguments for 'G':
%           A row or column vector with 18 numbers stating the Insertion Gain in dB in bands 1 through 18.
%           If this identifier is omitted, a default Insertion Gain of 0 dB is assumed in all 18 bands.
%
%   Arguments for 'T':
%           A row or column vector with 18 numbers stating the Hearing Threshold Levels in dBHL in bands 1 through 18.
%           If this identifier is omitted, a default Equivalent Hearing Threshold Level of 0 dBHL is assumed in all 18 bands .
%
%   Arguments for 'b':
%           A scalar, having a value of either 1 or 2, indicating the listening mode: b = 1 --> monaural listening, b = 2 --> binaural listening
%           If this identifier is omitted, monaural listening is assumed.
%
% Copyright 2005 Hannes Muesch & Pat Zurek

[x,Nvar]    = size(varargin);
CharCount   = 0;
Ident       = [];
for k = 1:Nvar
    if ischar(varargin{k})&(length(varargin{k})==1)
        CharCount = CharCount + 1;
        Ident = [Ident; k];
    end
end

if Nvar/CharCount ~= 2
    error('Every input must be preceeded by an identifying string')
else
    for n = 1:length(Ident)
        if upper(varargin{Ident(n)})     == 'G' % Insertion Gain [dB] (3.28)
            G = varargin{Ident(n)+1};
        elseif upper(varargin{Ident(n)}) == 'P' % CSNSL (3.17)
            P = varargin{Ident(n)+1};
        elseif upper(varargin{Ident(n)}) == 'M' % MTFI  (3.31)
            M = varargin{Ident(n)+1};
        elseif upper(varargin{Ident(n)}) == 'T' % Hearing Threshold Level
            T = varargin{Ident(n)+1};
        elseif upper(varargin{Ident(n)}) == 'B' % Monaural (1) or Binaural (2) listening ?
            B = varargin{Ident(n)+1};
        else
            error('Only ''G'', ''P'', ''M'', ''T'', and ''b'' are valid identifiers');
        end;
    end;
end;

if length(P) ~= 18,         error('Combined Speech and Noise Spectrum Level: Vector size incorrect');               end;
P = P(:)';
if any( size(M) - [18 9] ), error('Modulation Transfer Function for Intensity: Matrix size incorrect');             end;

if isempty(who('G')),
    G = zeros(1,18);
else
    if length(G) ~= 18,
        error('Insertion Gain: Vector size incorrect');
    end;
    G = G(:)';
end

% DERIVE EQUIVALENT HEARING THRESHOLD LEVEL
if isempty(who('T'))
    T = zeros(1,18);
else
    if length(T) ~= 18,
        error('Hearing Threshold Level: Vector size incorrect');
    end;
    T = T(:)';
end

if isempty(who('B'))
    B = 1;                                  % Default to monaural listening
else
    if ~((B==1)|(B==2))
        error('Invalid value of ''b'' specified!')
    end
end

if B==2                                     % Binaural listening
    T = T - 1.7;                            % Section 5.1.5
end






R = 10*log10( (M+eps)./(1-M+eps) );                                         % apparent speech-to-noise ratio (5.2.3.5, Eq. 22)
R = min(15,max(-15,R));                                                     % limit to range -15 ... +15 dB
R = mean(R');                                                               % Average across modulation frequencies (5.2.3.6)
E = R + 10*log10( 10.^(P/10) ./ (1 + 10.^(R/10)) );                         % Equivalent Speech and Equivalent Noise spectra (5.2.3.8, Eq. 23
N = E - R;                                                                  % ... and Eq 24)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E = E + G;                                                                  % Note: Eq. 23 and 24 represent the Equivalent Speech and Equivalent Noise
N = N + G;                                                                  % spectra only if the insertion gain is 0dB. Provisions for insertion gain
                                                                            % are not explicitly shown in the standard. However, they are necessary and
                                                                            % are applied here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% EOF



