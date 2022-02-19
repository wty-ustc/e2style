clc; clear;

n = 10^6;    % size of the cover
h = 10;      % constraint height - default is 10 - drives the complexity/quality tradeof
wetness = 0.6;  % relative wetness of the channel
alpha = 0.5;    % relative payload on the dry pixels

wet = rand(n, 1) < wetness;
dn = n - sum(wet);  % number of dry pixels
m = floor(dn * alpha);

cover = uint8(rand(n, 1));
message = uint8(rand(m, 1));
profile = ones(n, 1);  % constant profile
profile(wet) = Inf; % Wet pixels are assigned a weight of infinity, so they are never flipped. 

tic;
[dist, stego] = st_embed(cover, message, profile, h);
fprintf('distortion per dry cover element = %f\n', dist / dn);
fprintf('            embedding efficiency = %f\n', alpha / (dist / dn));
fprintf('                      throughput = %1.1f Kbits/sec\n', n / toc() / 1024);

message2 = st_extract(stego, m, h); % extract message
if all(message == message2)
    disp('Message has been extracted correctly.');
else
    error('Some error occured in the extraction process.');
end