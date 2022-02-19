clc; clear;

n = 10^5;    % size of the cover
h = 10;      % constraint height - default is 10 - drives the complexity/quality tradeof

e = zeros(19, 1);   % embedding efficiency
inva = zeros(19, 1);

for i = 2:20    % inverse relative payload 1/alpha
    i
    m = floor(n / i);   % number of message bits
    cover = uint8(rand(n, 1));
    message = uint8(rand(m, 1));
    profile = 1:n;  % linear profile
    profile = profile(randperm(n)) / sum(profile);  % create a random permutation and normalize
 
    dist = st_embed(cover, message, profile, h);
    inva(i) = n / m;
    e(i) = m / (n * dist);    
end

plot(inva, e);
title('Embedding efficiency for linear profile');
xlabel('Inverse payload 1/\alpha');
ylabel('Embedding efficiency e');