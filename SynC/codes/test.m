    fileFolder=fullfile('../../ZSL/ZSL_DATA/train_seen');
    dirOutput=dir(fullfile(fileFolder,'*.mat'));
    fileNames={dirOutput.name};
    n = length(fileNames);
    Xtr=[];
    Ytr=[];
    class_order=[];
    for i = 1:n %19
        class_order=[class_order,i];
        pth=fullfile('../../ZSL/ZSL_DATA/train_seen',fileNames{i});
        load(pth);
        Xtr=[Xtr;double(features)];% ȡ����train�� 19*   1300 * 2048 ������
        nn=size(features);
        nn=nn(1);
        idx=str2num(fileNames{i}(isstrprop(fileNames{i},'digit')));
        for j=1:nn
            Ytr=[Ytr,idx];% train-��Ӧ��ѵ������ǩ 19*  1300 \code_server\ZSL\ZSL_DATA\train_seen
        end
    end
        
    disp('load train ok')   
    %test
    fileFolder=fullfile('../../ZSL/ZSL_DATA/test/unseen');
    dirOutput=dir(fullfile(fileFolder,'*.mat'));
    fileNames={dirOutput.name};
    n = length(fileNames);
    Xte=[];
    Yte=[];
    for i = 1:n % 49
        pth=fullfile('../../ZSL/ZSL_DATA/test/unseen',fileNames{i});
        load(pth);
        
        nn=size(features);
        nn=nn(1); %dim 1
        if nn>50
            nn=50;
        end
        Xte=[Xte;double(features(1:nn,:))];% ȡ����train�� 49* <50 * 2048 ������
        idx=str2num(fileNames{i}(isstrprop(fileNames{i},'digit')));
        for j=1:nn
            Yte=[Yte,idx];% train-��Ӧ��ѵ������ǩ 19*  1300 \code_server\ZSL\ZSL_DATA\train_seen
        end
    end
    Yte=Yte';
    Ytr=Ytr';
    
    %Ytr = Ytr(:); % train-��Ӧ��ѵ������ǩ 19*  1300 \code_server\ZSL\ZSL_DATA\train_seen
    %Yte = Yte(:); % test-��Ӧ�Ĳ��Լ���ǩ 49* <50 = 2325 \code_server\ZSL\ZSL_DATA\test\unseen
    %X(isnan(X)) = 0;
    %X(isinf(X)) = 0;% train��test�� featrues���� (24700+2325) * 2048
    % X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 2))); % normalize
    %X(isnan(X)) = 0; X(isinf(X)) = 0;
    %Xte = X(te_loc, :);% ȡ����test��  2325 * 2048 ������
    %Xtr = X(tr_loc, :);% ȡ����test��  24700 * 2048 ������
    %clear X;
    fileFolder=fullfile('../../ZSL/ZSL_DATA/ImageNet/w2v.mat');
    load(fileFolder);
    attr2= w2v; % attr2-��Ӧw2v����
    
    disp('load test ok')
    clear features w2v wnids n nn fileNames pth dirOutput i j idx fileFolder