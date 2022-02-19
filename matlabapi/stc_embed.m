%STC_EMBED Embeds message into cover bitstring using Syndrome-Trellis Codes.
%
% [D Y] = STC_EMBED(X, MSG, COSTS) embeds message bit-string MSG into
% bit-string X. Cost of changing X(i) is COSTS(i). Use Syndrome-Trellis
% Codes with constraint height 10. Outputs stego bit-stream Y and total
% distortion D.
%
% [D Y] = STC_EMBED(X, MSG, COSTS, H) same as above, but uses STC of
% constraint height H.
%
% Input format:
%   X, MSG - vectors of type uint8
%   COSTS  - vector of type double
%   H      - scalar between 6 and 15 (a higher number means bigger 
%            efficiency but longer embedding time), default is 10.
%
% Use STC_EXTRACT(Y, H) to read the message.
%
% Author: Jan Judas
%
% STC Toolbox website: http://dde.binghamton.edu/filler/stc
%
% References:
% [1] T. Filler, J. Judas, J. Fridrich, "Minimizing Additive Distortion in 
%     Steganography using Syndrome-Trellis Codes", submitted to IEEE
%     Transactions on Information Forensics and Security, 2010.
%     http://dde.binghamton.edu/filler/pdf/Fill10tifs-stc.pdf
% 
% [2] T. Filler, J. Judas, J. Fridrich, "Minimizing Embedding Impact in 
%     Steganography using Trellis-Coded Quantization", Proc. SPIE,
%     Electronic Imaging, Media Forensics and Security XII, San Jose, CA, 
%     January 18-20, 2010.
%     http://dde.binghamton.edu/filler/pdf/Fill10spie-syndrome-trellis-codes.pdf
%
% [3] T. Filler, J. Fridrich, "Minimizing Additive Distortion Functions
%     With Non-Binary Embedding Operation in Steganography", 2nd IEEE 
%     Workshop on Information Forensics and Security, December 2010.
%     http://dde.binghamton.edu/filler/pdf/Fill10wifs-multi-layer-stc.pdf
%
%   See also STC_EXTRACT, STC_PM1_PLS_EMBED, STC_PM1_DLS_EMBED, 
%   STC_PM2_PLS_EMBED, STC_PM2_DLS_EMBED.
