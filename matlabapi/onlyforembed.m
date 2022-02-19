function x=onlyforembed(cover,message)
profile = ones(size(cover,1), 1);
[dist, stego] = stc_embed(cover, message, profile, 10);
x = stego;
end



