%data scaling
clear all;
clc;
load('AEMO_NSW.mat');

QDATA= TOTALDEMAND(2:end);
POINTS = [0,1488,2832,4320,5760,7248,8688,10176,11664,13104,14592,16032,17520];

MONTH = 1;
ODATA = QDATA(POINTS(MONTH)+1:POINTS(MONTH+2));
str = sprintf('NSW %d', MONTH);
disp(str);
DATA = (ODATA-min(ODATA))/(max(ODATA)-min(ODATA));
minvalue = min(ODATA);
scalesize = max(ODATA)-min(ODATA);

windowsize = 48;
samplesize = 48;
totalsize = length(DATA)-samplesize;

trainsize = floor(totalsize*3/48)*12;
testsize = floor((totalsize-trainsize)/12)*12;
totalsize = testsize+trainsize;

for i=1:totalsize;
    for j=1:windowsize;
    x(i,j)=DATA(i+j-1);
    end;
end;

for i=1:trainsize;
    for j=1:windowsize;
    train_x(i,j)=x(i,j);
    end;
end;

train_y=DATA(samplesize+1:trainsize+samplesize);
 
for i=1:testsize;
    for j=1:windowsize;
    test_x(i,j)=x(i+trainsize,j);
    end;
end;

test_y= DATA(trainsize+samplesize+1:trainsize+testsize+samplesize);




