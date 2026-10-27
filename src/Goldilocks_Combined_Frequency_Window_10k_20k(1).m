%% Goldilocks Experiment: Sampling Frequency x Observation Window x Dataset Size
% Estimates 3 fundamental peak amplitudes, 3 phase angles, and shared frequency.
% Requires MATLAB Deep Learning Toolbox. Not executed in MATLAB in this environment.
% Dataset sizes refer to total generated training/validation/internal-test examples.

clear; clc; close all;

%% 1. Experimental design
fsList = [1000 2000 5000 10000];              % Hz
windowMsList = [2 5 10 15 20];                % ms
datasetSizes = [10000 20000];                 % requested dataset sizes
trainingSeeds = 11;                            % use [11 22 33] for repeatability
nCommonTest = 1500;

VnomRMS = 220; Vpk = sqrt(2)*VnomRMS;
phaseNom = [0 -120 120];
ampRange = [0.90 1.10]; fRange = [49.5 50.5]; phaseDevMax = 5;
severeFraction = 0.20;
h5ModMax = 0.05; h7ModMax = 0.05;
h5SevRange = [0.05 0.10]; h7SevRange = [0.05 0.10];
snrMod = [30 40]; snrSev = [20 30]; dcModMax = 0.02; dcSevMax = 0.05;

hidden1 = 128; hidden2 = 64; miniBatch = 64;
initialLR = 3e-3; l2Reg = 5e-3; maxEpochs = 300;
lrDropPeriod = 100; valPatience = 40;
masterSeed = 42000; commonSeed = 900100;
outDir = 'Goldilocks_Combined_Results';
if ~exist(outDir,'dir'), mkdir(outDir); end

nConfigs = numel(fsList)*numel(windowMsList)*numel(datasetSizes)*numel(trainingSeeds);
fprintf('GOLDILOCKS COMBINED EXPERIMENT\n');
fprintf('Sampling frequencies: %s Hz\n',mat2str(fsList));
fprintf('Observation windows: %s ms\n',mat2str(windowMsList));
fprintf('Dataset sizes: %s\n',mat2str(datasetSizes));
fprintf('Total configurations per seed: %d\n',nConfigs);

allMetrics = table(); trainSummary = table(); commonPredictions = table();
configSummary = table(); configCount = 0;

%% 2. Run all frequency x window x dataset-size configurations
for wi = 1:numel(windowMsList)
    Tobs = windowMsList(wi)/1000;
    for fi = 1:numel(fsList)
        fs = fsList(fi);
        N = round(fs*Tobs);
        t = (0:N-1)/fs;
        nFeatures = 3*N;
        fprintf('\n====================================================\n');
        fprintf('Window %.1f ms | fs=%g Hz | N/phase=%d | inputs=%d\n',Tobs*1000,fs,N,nFeatures);
        if N < 7
            warning('This configuration has only %d samples per phase. Treat estimates as a diagnostic, not a practical candidate.',N);
        elseif N < 20
            fprintf('NOTE: Very short record; seven-parameter estimation may be poorly identifiable.\n');
        end

        % Data for this (fs, window) setting. Underlying parameter values are
        % reproducible by case index; noise samples vary with fs/window.
        [Xraw,Y,severity,meta] = makeData(max(datasetSizes),t,Vpk,ampRange,fRange, ...
            phaseNom,phaseDevMax,severeFraction,h5ModMax,h7ModMax,h5SevRange,h7SevRange, ...
            snrMod,snrSev,dcModMax,dcSevMax,masterSeed);
        [XtestRaw,Ytest,sevTest,metaTest] = makeData(nCommonTest,t,Vpk,ampRange,fRange, ...
            phaseNom,phaseDevMax,severeFraction,h5ModMax,h7ModMax,h5SevRange,h7SevRange, ...
            snrMod,snrSev,dcModMax,dcSevMax,commonSeed);

        % Fixed nominal scaling. DC offset remains present for the model to learn.
        X = Xraw/single(Vpk);
        Xcommon = XtestRaw/single(Vpk);

        % Nested split pools: larger datasets extend each split instead of
        % changing the examples already assigned to train/validation/test.
        rng(100000 + fs + round(Tobs*1e6),'twister');
        order = randperm(max(datasetSizes));
        maxM = max(datasetSizes);
        nTrPool = floor(0.70*maxM); nVaPool = floor(0.15*maxM);
        trPool = order(1:nTrPool);
        vaPool = order(nTrPool+1:nTrPool+nVaPool);
        tePool = order(nTrPool+nVaPool+1:end);

        for mi = 1:numel(datasetSizes)
            M = datasetSizes(mi);
            nTr=floor(.70*M); nVa=floor(.15*M); nTe=M-nTr-nVa;
            iTr=trPool(1:nTr); iVa=vaPool(1:nVa); iTe=tePool(1:nTe);
            Xtr=X(iTr,:); Xva=X(iVa,:); Xint=X(iTe,:);
            YtrRaw=Y(iTr,:); YvaRaw=Y(iVa,:);
            yMean=mean(YtrRaw,1); yStd=std(YtrRaw,0,1); yStd(yStd<1e-12)=1;
            Ytr=(YtrRaw-yMean)./yStd; Yva=(YvaRaw-yMean)./yStd;

            for si=1:numel(trainingSeeds)
                seed=trainingSeeds(si); configCount=configCount+1;
                fprintf('\n[%d/%d] fs=%g Hz | window=%.1f ms | M=%d | seed=%d\n', ...
                    configCount,nConfigs,fs,Tobs*1000,M,seed);
                layers=[
                    featureInputLayer(nFeatures,'Normalization','none','Name','input')
                    fullyConnectedLayer(hidden1,'Name','fc1')
                    reluLayer('Name','relu1')
                    fullyConnectedLayer(hidden2,'Name','fc2')
                    reluLayer('Name','relu2')
                    fullyConnectedLayer(7,'Name','output')
                    regressionLayer('Name','regression')];
                valFreq=max(5,ceil(nTr/miniBatch));
                rng(200000+fs+round(Tobs*1e6)+M+1000*seed,'twister');
                opts=trainingOptions('adam', ...
                    'MaxEpochs',maxEpochs,'MiniBatchSize',miniBatch, ...
                    'InitialLearnRate',initialLR,'LearnRateSchedule','piecewise', ...
                    'LearnRateDropPeriod',lrDropPeriod,'LearnRateDropFactor',0.25, ...
                    'L2Regularization',l2Reg,'Shuffle','every-epoch', ...
                    'ValidationData',{Xva,Yva},'ValidationFrequency',valFreq, ...
                    'ValidationPatience',valPatience,'OutputNetwork','best-validation-loss', ...
                    'Verbose',false,'Plots','training-progress','ExecutionEnvironment','auto');

                trainTimer=tic;
                [net,info]=trainNetwork(Xtr,Ytr,layers,opts);
                trainSec=toc(trainTimer);

                % Internal held-out test (from generated pool) and common test.
                YhatInt=double(predict(net,Xint,'MiniBatchSize',miniBatch)).*yStd+yMean;
                metInt=scoreOutputs(Y(iTe,:),YhatInt,severity(iTe),M,fs,Tobs*1000,"InternalTest",seed);
                allMetrics=[allMetrics;metInt]; %#ok<AGROW>
                Yhat=double(predict(net,Xcommon,'MiniBatchSize',miniBatch)).*yStd+yMean;
                met=scoreOutputs(Ytest,Yhat,sevTest,M,fs,Tobs*1000,"CommonTest",seed);
                allMetrics=[allMetrics;met]; %#ok<AGROW>

                bestVal=NaN; stopEpoch=NaN;
                if isfield(info,'ValidationLoss') && ~isempty(info.ValidationLoss)
                    v=info.ValidationLoss; v=v(~isnan(v)); if ~isempty(v), bestVal=min(v); end
                end
                if isfield(info,'Epoch') && ~isempty(info.Epoch), stopEpoch=max(info.Epoch); end
                trainSummary=[trainSummary;table(fs,Tobs*1000,M,N,nFeatures,nTr,nVa,nTe,seed,trainSec,stopEpoch,bestVal, ...
                    'VariableNames',{'SamplingFrequency_Hz','Window_ms','DatasetSize','SamplesPerPhase', ...
                    'InputFeatures','NumTraining','NumValidation','NumInternalTest','TrainingSeed', ...
                    'TrainingSeconds','StoppingEpoch','BestValidationLoss'})]; %#ok<AGROW>

                p=table(repmat(fs,nCommonTest,1),repmat(Tobs*1000,nCommonTest,1), ...
                    repmat(M,nCommonTest,1),repmat(seed,nCommonTest,1),(1:nCommonTest)',sevTest, ...
                    Ytest(:,1),Yhat(:,1),Ytest(:,2),Yhat(:,2),Ytest(:,3),Yhat(:,3), ...
                    Ytest(:,4),Yhat(:,4),Ytest(:,5),Yhat(:,5),Ytest(:,6),Yhat(:,6), ...
                    Ytest(:,7),Yhat(:,7), ...
                    'VariableNames',{'SamplingFrequency_Hz','Window_ms','DatasetSize','TrainingSeed','Case','Severity', ...
                    'TrueAmpA_Vpeak','PredAmpA_Vpeak','TrueAmpB_Vpeak','PredAmpB_Vpeak', ...
                    'TrueAmpC_Vpeak','PredAmpC_Vpeak','TruePhaseA_deg','PredPhaseA_deg', ...
                    'TruePhaseB_deg','PredPhaseB_deg','TruePhaseC_deg','PredPhaseC_deg', ...
                    'TrueFrequency_Hz','PredFrequency_Hz'});
                commonPredictions=[commonPredictions;p]; %#ok<AGROW>

                save(fullfile(outDir,sprintf('model_fs%gHz_win%gms_M%d_seed%d.mat',fs,Tobs*1000,M,seed)), ...
                    'net','info','yMean','yStd','Vpk','fs','Tobs','M','N','t','seed');
                fprintf('Training %.1f s | stopping epoch %g | best validation loss %.5g\n',trainSec,stopEpoch,bestVal);
                disp(met(met.Severity=="Overall",{'Output','Unit','MAE','RMSE'}));
            end
        end

        % Store case labels and disturbance metadata for reproducibility.
        labelNames={'AmpA_Vpeak','AmpB_Vpeak','AmpC_Vpeak','PhaseA_deg','PhaseB_deg','PhaseC_deg','Frequency_Hz'};
        lab=array2table(Y,'VariableNames',labelNames); lab.Severity=severity;
        lab=[table((1:height(lab))','VariableNames',{'Case'}) lab meta];
        writetable(lab,fullfile(outDir,sprintf('master_labels_fs%gHz_win%gms.csv',fs,Tobs*1000)));
        tlab=array2table(Ytest,'VariableNames',labelNames); tlab.Severity=sevTest;
        tlab=[table((1:height(tlab))','VariableNames',{'Case'}) tlab metaTest];
        writetable(tlab,fullfile(outDir,sprintf('common_test_labels_fs%gHz_win%gms.csv',fs,Tobs*1000)));
    end
end

%% 3. Save tables and primary common-test results
writetable(allMetrics,fullfile(outDir,'all_error_metrics.csv'));
writetable(trainSummary,fullfile(outDir,'training_summary.csv'));
writetable(commonPredictions,fullfile(outDir,'common_test_predictions.csv'));
main=allMetrics(allMetrics.TestSet=="CommonTest" & allMetrics.Severity=="Overall",:);
writetable(main,fullfile(outDir,'main_common_test_results.csv'));

% Pivot-friendly overall summary: each row is one configuration/output.
[G,gfs,gwin,gM,gOut,gUnit]=findgroups(main.SamplingFrequency_Hz,main.Window_ms,main.DatasetSize,main.Output,main.Unit);
meanMAE=splitapply(@mean,main.MAE,G); stdMAE=splitapply(@std,main.MAE,G);
meanRMSE=splitapply(@mean,main.RMSE,G); stdRMSE=splitapply(@std,main.RMSE,G);
summary=table(gfs,gwin,gM,gOut,gUnit,meanMAE,stdMAE,meanRMSE,stdRMSE, ...
    'VariableNames',{'SamplingFrequency_Hz','Window_ms','DatasetSize','Output','Unit','MeanMAE','StdMAE','MeanRMSE','StdRMSE'});
writetable(summary,fullfile(outDir,'common_test_summary.csv'));

settings=struct('fsList',fsList,'windowMsList',windowMsList,'datasetSizes',datasetSizes, ...
    'trainingSeeds',trainingSeeds,'nCommonTest',nCommonTest,'VnomRMS',VnomRMS,'Vpk',Vpk, ...
    'ampRange',ampRange,'fRange',fRange,'phaseDevMax',phaseDevMax,'severeFraction',severeFraction, ...
    'h5ModMax',h5ModMax,'h7ModMax',h7ModMax,'h5SevRange',h5SevRange,'h7SevRange',h7SevRange, ...
    'snrMod',snrMod,'snrSev',snrSev,'dcModMax',dcModMax,'dcSevMax',dcSevMax, ...
    'hiddenUnits',[hidden1 hidden2],'miniBatch',miniBatch,'initialLR',initialLR,'l2Reg',l2Reg, ...
    'maxEpochs',maxEpochs,'lrDropPeriod',lrDropPeriod,'valPatience',valPatience, ...
    'preprocessing','fixed nominal-peak scaling; DC offset retained');
save(fullfile(outDir,'complete_results.mat'),'allMetrics','trainSummary','commonPredictions','main','summary','settings');

%% 4. Visualize results: MAE and RMSE against observation window, one curve per fs
outputs=["Amplitude A","Amplitude B","Amplitude C","Phase A","Phase B","Phase C","Frequency"];
units=["V peak","V peak","V peak","deg","deg","deg","Hz"];
for metricName=["MeanMAE","MeanRMSE"]
    for M=datasetSizes
        fig=figure('Name',sprintf('%s_M%d',metricName,M),'Color','w','Position',[60 60 1400 800]);
        tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
        for oi=1:numel(outputs)
            nexttile; hold on; grid on; box on;
            for fi=1:numel(fsList)
                fs=fsList(fi);
                rows=summary(summary.SamplingFrequency_Hz==fs & summary.DatasetSize==M & summary.Output==outputs(oi),:);
                [~,ord]=sort(rows.Window_ms);
                yvals=rows.(char(metricName));
                plot(rows.Window_ms(ord),yvals(ord),'-o','LineWidth',1.5, ...
                    'DisplayName',sprintf('%g kHz',fs/1000));
            end
            title(outputs(oi)); xlabel('Observation window (ms)');
            ylabel(sprintf('%s (%s)',metricName,units(oi))); legend('Location','best');
        end
        sgtitle(sprintf('Common-test %s | Dataset size = %d',metricName,M));
        exportgraphics(fig,fullfile(outDir,sprintf('Figure_%s_M%d.png',metricName,M)),'Resolution',200);
        savefig(fig,fullfile(outDir,sprintf('Figure_%s_M%d.fig',metricName,M)));
    end
end

% One concise table with mean MAE over the six amplitude/phase outputs and frequency.
% This is a descriptive ranking aid only: report each physical-unit metric separately too.
configs=unique(main(:,{'SamplingFrequency_Hz','Window_ms','DatasetSize'}));
configs.MeanAmplitudeMAE_Vpeak=zeros(height(configs),1);
configs.MeanPhaseMAE_deg=zeros(height(configs),1);
configs.FrequencyMAE_Hz=zeros(height(configs),1);
configs.TotalTrainingSeconds=zeros(height(configs),1);
for k=1:height(configs)
    mask=main.SamplingFrequency_Hz==configs.SamplingFrequency_Hz(k) & ...
        main.Window_ms==configs.Window_ms(k) & main.DatasetSize==configs.DatasetSize(k);
    r=main(mask,:);
    freqMask=r.Output=="Frequency";
    configs.MeanAmplitudeMAE_Vpeak(k)=mean(r.MAE(contains(r.Output,"Amplitude")));
    configs.MeanPhaseMAE_deg(k)=mean(r.MAE(contains(r.Output,"Phase")));
    configs.FrequencyMAE_Hz(k)=r.MAE(freqMask);
    configs.TotalTrainingSeconds(k)=sum(trainSummary.TrainingSeconds(trainSummary.SamplingFrequency_Hz==configs.SamplingFrequency_Hz(k) & ...
        trainSummary.Window_ms==configs.Window_ms(k) & trainSummary.DatasetSize==configs.DatasetSize(k)));
end
writetable(configs,fullfile(outDir,'configuration_comparison.csv'));

fprintf('\nFinished. Results saved in: %s\n',outDir);
fprintf('Primary results: main_common_test_results.csv\n');
fprintf('Configuration comparison: configuration_comparison.csv\n');
fprintf('This script was not executed in MATLAB here; check version-specific toolbox/API compatibility.\n');

%% Local functions
function [X,Y,severity,meta]=makeData(n,t,Vpk,ampRange,fRange,phaseNom,phaseDevMax, ...
    severeFraction,h5ModMax,h7ModMax,h5SevRange,h7SevRange,snrMod,snrSev,dcModMax,dcSevMax,baseSeed)
N=numel(t); X=zeros(n,3*N,'single'); Y=zeros(n,7); severity=strings(n,1);
meta=table('Size',[n 6],'VariableTypes',repmat({'double'},1,6), ...
    'VariableNames',{'SNR_dB','H5MaxFraction','H7MaxFraction','DCMaxFraction','FrequencyDeviation_Hz','SevereFlag'});
for ii=1:n
    rng(baseSeed+ii,'twister');
    A=Vpk*(ampRange(1)+(ampRange(2)-ampRange(1))*rand(1,3));
    f=fRange(1)+(fRange(2)-fRange(1))*rand();
    phaseDeg=phaseNom+(2*rand(1,3)-1)*phaseDevMax; phaseRad=deg2rad(phaseDeg);
    isSevere=rand()<severeFraction;
    if isSevere
        h5=h5SevRange(1)+(h5SevRange(2)-h5SevRange(1))*rand(1,3);
        h7=h7SevRange(1)+(h7SevRange(2)-h7SevRange(1))*rand(1,3);
        snrDb=snrSev(1)+(snrSev(2)-snrSev(1))*rand(); dcFrac=dcSevMax*rand(1,3); severity(ii)="Severe";
    else
        h5=h5ModMax*rand(1,3); h7=h7ModMax*rand(1,3);
        snrDb=snrMod(1)+(snrMod(2)-snrMod(1))*rand(); dcFrac=dcModMax*rand(1,3); severity(ii)="Moderate";
    end
    dc=(2*rand(1,3)-1).*dcFrac.*A;
    h5phase=2*pi*rand(1,3); h7phase=2*pi*rand(1,3); wave=zeros(3,N);
    for pp=1:3
        fundamental=A(pp)*sin(2*pi*f*t+phaseRad(pp));
        fifth=A(pp)*h5(pp)*sin(2*pi*5*f*t+h5phase(pp));
        seventh=A(pp)*h7(pp)*sin(2*pi*7*f*t+h7phase(pp));
        ac=fundamental+fifth+seventh;
        noiseRMS=sqrt(mean(ac.^2))/(10^(snrDb/20));
        wave(pp,:)=ac+dc(pp)+noiseRMS*randn(1,N);
    end
    X(ii,:)=single(reshape(wave.',1,[])); Y(ii,:)=[A phaseDeg f];
    meta.SNR_dB(ii)=snrDb; meta.H5MaxFraction(ii)=max(h5); meta.H7MaxFraction(ii)=max(h7);
    meta.DCMaxFraction(ii)=max(dcFrac); meta.FrequencyDeviation_Hz(ii)=f-50; meta.SevereFlag(ii)=double(isSevere);
end
end

function metrics=scoreOutputs(Ytrue,Ypred,severity,M,fs,windowMs,testSet,seed)
names=["Amplitude A";"Amplitude B";"Amplitude C";"Phase A";"Phase B";"Phase C";"Frequency"];
units=["V peak";"V peak";"V peak";"deg";"deg";"deg";"Hz"];
groups=["Overall";"Moderate";"Severe"]; rows=cell(0,10);
for gi=1:numel(groups)
    if groups(gi)=="Overall", mask=true(size(severity)); else, mask=severity==groups(gi); end
    if ~any(mask), continue; end
    for j=1:7
        e=Ypred(mask,j)-Ytrue(mask,j);
        if j>=4 && j<=6, e=mod(e+180,360)-180; end
        rows(end+1,:)={fs,windowMs,M,string(testSet),groups(gi),names(j),units(j),mean(abs(e)),sqrt(mean(e.^2)),seed}; %#ok<AGROW>
    end
end
metrics=cell2table(rows,'VariableNames',{'SamplingFrequency_Hz','Window_ms','DatasetSize','TestSet','Severity','Output','Unit','MAE','RMSE','TrainingSeed'});
end
