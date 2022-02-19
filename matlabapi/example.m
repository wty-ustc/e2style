%clc; clear;

n = 10^5;    % size of the cover
alpha = 0.5; % relative payload
h = 10;      % constraint height - default is 10 - drives the complexity/quality tradeof

cover = uint8(rand(n, 1));
m = round(n * alpha); % number of message bits
message = uint8(rand(m, 1));
profile = ones(n, 1);

tic;
[dist, stego] = stc_embed(cover, message, profile, h); % embed message

fprintf('distortion per cover element = %f\n', dist / n);
fprintf('        embedding efficiency = %f\n', alpha / (dist / n));
fprintf('                  throughput = %1.1f Kbits/sec\n', n / toc() / 1024);

message2 = stc_extract(stego, m, h); % extract message
if all(message == message2)
    disp('Message has been extracted correctly.');
else
    error('Some error occured in the extraction process.');
end