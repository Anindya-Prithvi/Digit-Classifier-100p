function [img] = processimg(k)
img=im2double(imread(k));
img=img(:);
img=img';
img=[1-img];
img=img.*2^10;
end
