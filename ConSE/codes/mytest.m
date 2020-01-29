fileFolder=fullfile('../../ZSL/ZSL_DATA/test/combine');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
n = length(fileNames);
Xte=[];
Yte=[];
file=[];
for i = 1:n % 49
    idx=str2num(fileNames{i}(isstrprop(fileNames{i},'digit')));
    file=[file idx];
    file=sort(file);
end
for i = 1:n % 49
    %disp(file(i))
    tmp=[num2str(file(i)),'.mat'];
    pth=fullfile('../../ZSL/ZSL_DATA/test/combine',tmp);
    load(pth);
    nn=size(features);
    nn=nn(1); %dim 1
    if nn>50
        nn=50;
    end
    Xte=[Xte;double(features(1:nn,:))];% 取出来train的 49* <50 * 2048 的数据

    for j=1:nn
        Yte=[Yte,file(i)];% train-对应的训练集标签 19*  1300 \code_server\ZSL\ZSL_DATA\train_seen
    end
end
Yte=Yte';