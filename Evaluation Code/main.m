clear; close all; clc;

%SaliencyMap Path setting
SalMapPath = '../SalMap/'; %Put model results in this folder.
Models = {'EGNet_ResNet50'};% You can add other model like: Models = {'13MR','16LDW','17SPMIL'};
modelNum = length(Models);

%Datasets setting
DataPath = 'E:/2sal/dataset/CoSOD/';
Datasets = {'MSRC','iCoseg'};% You may also need other datasets, such as Datasets = {'Image Pair','MSRC','iCoseg','Cosal2015'};

%Results setting
ResDir = '../Result/';

Thresholds = 1:-1/255:0;

datasetNum = length(Datasets);
    
for d = 1:datasetNum
    
    tic;
    dataset = Datasets{d};
    fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
    
    ResPath = [ResDir dataset '-mat/']; %The result will be saved in *.mat file so that you can used it for the next time.
    if ~exist(ResPath,'dir')
        mkdir(ResPath);
    end
    resTxt = [ResDir dataset '_result.txt'];  %The evaluation result will be saved in ../Result folder.
    fileID = fopen(resTxt,'w');
    
    for m = 1:modelNum
        model = Models{m};
        
        gtGroupPath = [DataPath dataset '/groundtruth/'];
        gtGroup = dir(gtGroupPath);
        
        for imgGroupIndex=3:length(gtGroup)
            GroupNameStr = gtGroup(imgGroupIndex).name;
            gtPath = [DataPath dataset '/groundtruth/' GroupNameStr '/'];
            salPath = [SalMapPath model '/' dataset '/' GroupNameStr '/'];
            
            imgFiles = dir([salPath '*_PSFEM_3.png']);
            imgNUM = length(imgFiles);

            [threshold_Fmeasure, threshold_Emeasure] = deal(zeros(imgNUM,length(Thresholds)));

            [threshold_Precion, threshold_Recall] = deal(zeros(imgNUM,length(Thresholds)));

            [Smeasure, adpFmeasure, adpEmeasure, MAE] =deal(zeros(1,imgNUM));
        
            parfor i = 1:imgNUM

                fprintf('Evaluating(%s Dataset,%s Group,%s Model): %d/%d\n',dataset, GroupNameStr, model, i,imgNUM);
                name =  imgFiles(i).name;

                %load gt
                gt = imread([gtPath name(1:end-12) '.png']);


                if (ndims(gt)>2)
                    gt = rgb2gray(gt);
                end

                if ~islogical(gt)
                    gt = gt(:,:,1) > 128;
                end

                %load salency
                sal  = imread([salPath name]);

                %check size
                if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
                    sal = imresize(sal,size(gt));
                    imwrite(sal,[salPath name]);
                    fprintf('Error occurs in the path: %s!!!\n', [salPath name]);
                end

                sal = im2double(sal(:,:,1));

                %normalize sal to [0, 1]
                sal = reshape(mapminmax(sal(:)',0,1),size(sal));
                Smeasure(i) = StructureMeasure(sal,logical(gt));

                % Using the 2 times of average of sal map as the threshold.
                threshold =  2* mean(sal(:)) ;
                [~,~,adpFmeasure(i)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);


                Bi_sal = zeros(size(sal));
                Bi_sal(sal>threshold)=1;
                adpEmeasure(i) = Enhancedmeasure(Bi_sal,gt);

                [threshold_F, threshold_E]  = deal(zeros(1,length(Thresholds)));
                [threshold_Pr, threshold_Rec]  = deal(zeros(1,length(Thresholds)));
                for t = 1:length(Thresholds)
                    threshold = Thresholds(t);
                    [threshold_Pr(t), threshold_Rec(t), threshold_F(t)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);

                    Bi_sal = zeros(size(sal));
                    Bi_sal(sal>threshold)=1;
                    threshold_E(t) = Enhancedmeasure(Bi_sal,gt);
                end

                threshold_Fmeasure(i,:) = threshold_F;
                threshold_Emeasure(i,:) = threshold_E;
                threshold_Precion(i,:) = threshold_Pr;
                threshold_Recall(i,:) = threshold_Rec;

                MAE(i) = mean2(abs(double(logical(gt)) - sal));

            end
            
            if imgGroupIndex == 3
                % group calc init
                GroupThreshold_Fmeasure = threshold_Fmeasure;
                GroupThreshold_Emeasure = threshold_Emeasure;
                GroupThreshold_Precion = threshold_Precion;
                GroupThreshold_Recall = threshold_Recall;
                GroupMAE = MAE;
                GroupSmeasure = Smeasure;
                GroupadpEmeasure = adpEmeasure;
                GroupadpFmeasure = adpFmeasure;
            else
                GroupThreshold_Fmeasure = vertcat(GroupThreshold_Fmeasure,threshold_Fmeasure);
                GroupThreshold_Emeasure = vertcat(GroupThreshold_Emeasure,threshold_Emeasure);
                GroupThreshold_Precion = vertcat(GroupThreshold_Precion,threshold_Precion);
                GroupThreshold_Recall = vertcat(GroupThreshold_Recall,threshold_Recall);
                GroupMAE = horzcat(GroupMAE,MAE);
                GroupSmeasure = horzcat(GroupSmeasure,Smeasure);
                GroupadpEmeasure = horzcat(GroupadpEmeasure,adpEmeasure);
                GroupadpFmeasure = horzcat(GroupadpEmeasure,adpFmeasure);
            end
        
        end
        column_F = mean(GroupThreshold_Fmeasure,1); % 在各个维度对所有的img进行度量
        meanFm = mean(column_F);    % 计算各个维度的均值
        maxFm = max(column_F);  % 计算各个维度的最大值
        
        column_Pr = mean(GroupThreshold_Precion,1);
        column_Rec = mean(GroupThreshold_Recall,1);
        
        column_E = mean(GroupThreshold_Emeasure,1);
        meanEm = mean(column_E);
        maxEm = max(column_E);
        
        Smeasure = mean2(GroupSmeasure);
        adpFm = mean2(GroupadpFmeasure);
        adpEm = mean2(GroupadpEmeasure);
        mae = mean2(GroupMAE);
        
        save([ResPath model],'Smeasure', 'mae', 'column_Pr', 'column_Rec', 'column_F', 'adpFm', 'meanFm', 'maxFm', 'column_E', 'adpEm', 'meanEm', 'maxEm');
        fprintf(fileID, '(Dataset:%s; Model:%s) maxFm:%.3f; MAE:%.3f; Smeasure:%.3f; maxEm:%.3f; meanEm:%.3f; adpEm:%.3f; meanFm:%.3f; adpFm:%.3f.\n', dataset, model, maxFm, mae, Smeasure, maxEm, meanEm, adpEm, meanFm, adpFm); 
    end
    toc;
    
end