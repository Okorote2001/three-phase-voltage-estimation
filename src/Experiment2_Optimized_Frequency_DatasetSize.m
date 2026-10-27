%% Optimized Experiment 2: Sampling frequency x dataset size
% MATLAB Deep Learning Toolbox required.
% Outputs: 3 peak amplitudes (V), 3 phases (deg), common frequency (Hz).
% This is a simulation experiment; validate against real measurements before deployment.

clear; clc; close all;

%% Experimental settings
fsList = [1000 2000 5000 10000];
datasetSizes = [2000 5000 10000 15000 20000];
Tobs = 0.020;
VnomRMS = 220; Vpk = sqrt(2)*VnomRMS;
phaseNom = [0 -120 120];
ampRange = [0.90 1.10]; fRange = [49.5 50.5]; phaseDevMax = 5;
severeFraction = 0.20;
h5ModMax = 0.05; h7ModMax = 0.05;
h5SevRange = [0.05 0.10]; h7SevRange = [0.05 0.10];
snrMod = [30 40]; snrSev = [20 30]; dcModMax = 0.02; dcSevMax = 0.05;

hidden1 = 128; hidden2 = 64; miniBatch = 64;
initialLR = 3e-3; l2Reg = 5e-3; maxEpochs = 300;
lrDropPeriod = 100; valPatience = 40; nCommonTest = 1500;
% Set [11 22 33] or more seeds for repeatability analysis. Start with [11] for a first run.
trainingSeeds = [11];
masterSeed = 42000; commonSeed = 900100;
outDir = 'Experiment2_Optimized_Frequency_DatasetSize_Results';
if ~exist(outDir,'dir'), mkdir(outDir); end

fprintf('Optimized frequency x dataset-size experiment\n');
fprintf('Tobs = %.1f ms | fs = %s | sizes = %s\n',Tobs*1000,mat2str(fsList),mat2str(datasetSizes));
fprintf('Training seeds: %s\n',mat2str(trainingSeeds));

allMetrics = table(); trainSummary = table(); commonPredictions = table();

for fi = 1:numel(fsList)
    fs = fsList(fi);
    N = round(fs*Tobs);             % exact sample count; avoid floating colon issues
    t = (0:N-1)/fs;
    nFeatures = 3*N;
    fprintf('\n=== fs=%g Hz | N/phase=%d | features=%d ===\n',fs,N,nFeatures);

    % Matched case seeds give the same underlying labels/disturbance parameters
    % across fs. Noise samples are regenerated for each sampled waveform.
    [Xraw,Y,severity,meta] = makeData(max(datasetSizes),t,Vpk,ampRange,fRange, ...
        phaseNom,phaseDevMax,severeFraction,h5ModMax,h7ModMax,h5SevRange,h7SevRange, ...
        snrMod,snrSev,dcModMax,dcSevMax,masterSeed);
    [XtestRaw,Ytest,sevTest,metaTest] = makeData(nCommonTest,t,Vpk,ampRange,fRange, ...
        phaseNom,phaseDevMax,severeFraction,h5ModMax,h7ModMax,h5SevRange,h7SevRange, ...
        snrMod,snrSev,dcModMax,dcSevMax,commonSeed);

    % Fixed scaling only. DC remains in the input, so the NN must learn robustness to it.
    X = Xraw/single(Vpk); Xcommon = XtestRaw/single(Vpk);

    % Create nested split pools once. The 5k dataset includes the same first
    % 2k examples by partition; larger sizes extend, rather than reshuffle, splits.
    rng(100000+fs,'twister'); order = randperm(max(datasetSizes));
    maxM = max(datasetSizes); nTrPool=floor(.70*maxM); nVaPool=floor(.15*maxM);
    trPool=order(1:nTrPool); vaPool=order(nTrPool+1:nTrPool+nVaPool);
    tePool=order(nTrPool+nVaPool+1:end);

    for mi = 1:numel(datasetSizes)
        M = datasetSizes(mi);
        nTr=floor(.70*M); nVa=floor(.15*M); nTe=M-nTr-nVa;
        iTr=trPool(1:nTr); iVa=vaPool(1:nVa); iTe=tePool(1:nTe);
        Xtr=X(iTr,:); Xva=X(iVa,:); Xint=X(iTe,:);
        YtrRaw=Y(iTr,:); YvaRaw=Y(iVa,:);
        yMean=mean(YtrRaw,1); yStd=std(YtrRaw,0,1); yStd(yStd<1e-12)=1;
        Ytr=(YtrRaw-yMean)./yStd; Yva=(YvaRaw-yMean)./yStd;

        for si=1:numel(trainingSeeds)
            seed=trainingSeeds(si);
            fprintf('\n--- fs=%g Hz | M=%d | seed=%d ---\n',fs,M,seed);
            layers=[
                featureInputLayer(nFeatures,'Normalization','none','Name','input')
                fullyConnectedLayer(hidden1,'Name','fc1')
                reluLayer('Name','relu1')
                fullyConnectedLayer(hidden2,'Name','fc2')
                reluLayer('Name','relu2')
                fullyConnectedLayer(7,'Name','output')
                regressionLayer('Name','regression')];
            valFreq=max(5,ceil(nTr/miniBatch));
            rng(200000+fs+M+1000*seed,'twister');
            opts=trainingOptions('adam', ...
                'MaxEpochs',maxEpochs,'MiniBatchSize',miniBatch, ...
                'InitialLearnRate',initialLR,'LearnRateSchedule','piecewise', ...
                'LearnRateDropPeriod',lrDropPeriod,'LearnRateDropFactor',0.25, ...
                'L2Regularization',l2Reg,'Shuffle','every-epoch', ...
                'ValidationData',{Xva,Yva},'ValidationFrequency',valFreq, ...
                'ValidationPatience',valPatience,'OutputNetwork','best-validation-loss', ...
                'Verbose',false,'Plots','training-progress','ExecutionEnvironment','auto');
            ticTrain=tic; [net,info]=trainNetwork(Xtr,Ytr,layers,opts); trainSec=toc(ticTrain);

            YhatInt=double(predict(net,Xint,'MiniBatchSize',miniBatch)).*yStd+yMean;
            metInt=scoreOutputs(Y(iTe,:),YhatInt,severity(iTe),M,fs,"InternalTest",seed);
            allMetrics=[allMetrics;metInt]; %#ok<AGROW>
            Yhat=double(predict(net,Xcommon,'MiniBatchSize',miniBatch)).*yStd+yMean;
            met=scoreOutputs(Ytest,Yhat,sevTest,M,fs,"CommonTest",seed);
            allMetrics=[allMetrics;met]; %#ok<AGROW>

            bestVal=NaN; stopEpoch=NaN;
            if isfield(info,'ValidationLoss') && ~isempty(info.ValidationLoss)
                v=info.ValidationLoss; v=v(~isnan(v)); if ~isempty(v), bestVal=min(v); end
            end
            if isfield(info,'Epoch') && ~isempty(info.Epoch), stopEpoch=max(info.Epoch); end
            trainSummary=[trainSummary;table(fs,Tobs,M,N,nFeatures,nTr,nVa,nTe,seed,trainSec,stopEpoch,bestVal, ...
                'VariableNames',{'SamplingFrequency_Hz','Window_s','DatasetSize','SamplesPerPhase', ...
                'InputFeatures','NumTraining','NumValidation','NumInternalTest','TrainingSeed', ...
                'TrainingSeconds','StoppingEpoch','BestValidationLoss'})]; %#ok<AGROW>

            p=table(repmat(fs,nCommonTest,1),repmat(M,nCommonTest,1),repmat(seed,nCommonTest,1), ...
                (1:nCommonTest)',sevTest,Ytest(:,1),Yhat(:,1),Ytest(:,2),Yhat(:,2),Ytest(:,3),Yhat(:,3), ...
                Ytest(:,4),Yhat(:,4),Ytest(:,5),Yhat(:,5),Ytest(:,6),Yhat(:,6),Ytest(:,7),Yhat(:,7), ...
                'VariableNames',{'SamplingFrequency_Hz','DatasetSize','TrainingSeed','Case','Severity', ...
                'TrueAmpA_Vpeak','PredAmpA_Vpeak','TrueAmpB_Vpeak','PredAmpB_Vpeak', ...
                'TrueAmpC_Vpeak','PredAmpC_Vpeak','TruePhaseA_deg','PredPhaseA_deg', ...
                'TruePhaseB_deg','PredPhaseB_deg','TruePhaseC_deg','PredPhaseC_deg', ...
                'TrueFrequency_Hz','PredFrequency_Hz'});
            commonPredictions=[commonPredictions;p]; %#ok<AGROW>
            save(fullfile(outDir,sprintf('model_%gHz_%d_examples_seed%d.mat',fs,M,seed)), ...
                'net','info','yMean','yStd','Vpk','fs','Tobs','M','t','seed');
            fprintf('Training %.1f s | stopping epoch %g | best validation loss %.5g\n',trainSec,stopEpoch,bestVal);
            disp(met(met.Severity=="Overall",{'Output','Unit','MAE','RMSE'}));
        end
    end

    % Save explicit data-generation metadata for reproducibility.
    labelNames={'AmpA_Vpeak','AmpB_Vpeak','AmpC_Vpeak','PhaseA_deg','PhaseB_deg','PhaseC_deg','Frequency_Hz'};
    lab=array2table(Y,'VariableNames',labelNames); lab.Severity=severity;
    lab=[table((1:height(lab))','VariableNames',{'Case'}) lab meta];
    writetable(lab,fullfile(outDir,sprintf('master_labels_%gHz.csv',fs)));
    tlab=array2table(Ytest,'VariableNames',labelNames); tlab.Severity=sevTest;
    tlab=[table((1:height(tlab))','VariableNames',{'Case'}) tlab metaTest];
    writetable(tlab,fullfile(outDir,sprintf('common_test_labels_%gHz.csv',fs)));
end

writetable(allMetrics,fullfile(outDir,'all_error_metrics.csv'));
writetable(trainSummary,fullfile(outDir,'training_summary.csv'));
writetable(commonPredictions,fullfile(outDir,'common_test_predictions.csv'));
main=allMetrics(allMetrics.TestSet=="CommonTest" & allMetrics.Severity=="Overall",:);
writetable(main,fullfile(outDir,'main_common_test_results.csv'));

% Summarize variability across training seeds. With one seed, SD is NaN by definition.
[G,gfs,gM,gOut,gUnit]=findgroups(main.SamplingFrequency_Hz,main.DatasetSize,main.Output,main.Unit);
meanMAE=splitapply(@mean,main.MAE,G); stdMAE=splitapply(@std,main.MAE,G);
meanRMSE=splitapply(@mean,main.RMSE,G); stdRMSE=splitapply(@std,main.RMSE,G);
seedSummary=table(gfs,gM,gOut,gUnit,meanMAE,stdMAE,meanRMSE,stdRMSE, ...
    'VariableNames',{'SamplingFrequency_Hz','DatasetSize','Output','Unit','MeanMAE','StdMAE','MeanRMSE','StdRMSE'});
writetable(seedSummary,fullfile(outDir,'common_test_seed_summary.csv'));
settings=struct('fsList',fsList,'datasetSizes',datasetSizes,'Tobs',Tobs,'trainingSeeds',trainingSeeds, ...
    'split',[.70 .15 .15],'nCommonTest',nCommonTest,'VnomRMS',VnomRMS,'Vpk',Vpk, ...
    'ampRange',ampRange,'fRange',fRange,'phaseDevMax',phaseDevMax,'severeFraction',severeFraction, ...
    'h5ModMax',h5ModMax,'h7ModMax',h7ModMax,'h5SevRange',h5SevRange,'h7SevRange',h7SevRange, ...
    'snrMod',snrMod,'snrSev',snrSev,'dcModMax',dcModMax,'dcSevMax',dcSevMax, ...
    'hiddenUnits',[hidden1 hidden2],'miniBatch',miniBatch,'initialLR',initialLR,'l2Reg',l2Reg, ...
    'maxEpochs',maxEpochs,'lrDropPeriod',lrDropPeriod,'valPatience',valPatience, ...
    'preprocessing','fixed nominal-peak scaling; DC offset retained');
save(fullfile(outDir,'complete_results.mat'),'allMetrics','trainSummary','commonPredictions','main','seedSummary','settings');

% Summary plots (MAE and RMSE) for all seven outputs.
outputs=["Amplitude A","Amplitude B","Amplitude C","Phase A","Phase B","Phase C","Frequency"];
units=["V peak","V peak","V peak","deg","deg","deg","Hz"];
for metricName=["MeanMAE","MeanRMSE"]
    fig=figure('Name',char(metricName),'Color','w','Position',[60 60 1400 800]);
    tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
    for oi=1:numel(outputs)
        nexttile; hold on; grid on; box on;
        for fi=1:numel(fsList)
            fs=fsList(fi); rows=seedSummary(seedSummary.SamplingFrequency_Hz==fs & seedSummary.Output==outputs(oi),:);
            [~,ord]=sort(rows.DatasetSize);
            yvals = rows.(char(metricName));
            plot(rows.DatasetSize(ord),yvals(ord),'-o','LineWidth',1.5, ...
                'DisplayName',sprintf('%g kHz',fs/1000));
            if numel(trainingSeeds)>1
                errCol='StdMAE'; if metricName=="MeanRMSE", errCol='StdRMSE'; end
                errorbar(rows.DatasetSize(ord),rows.(char(metricName))(ord),rows.(errCol)(ord), ...
                    'LineStyle','none','HandleVisibility','off');
            end
        end
        title(outputs(oi)); xlabel('Dataset size (examples)');
        ylabel(sprintf('%s (%s)',metricName,units(oi))); legend('Location','best');
    end
    sgtitle(sprintf('Common-test %s vs Dataset Size and Sampling Frequency',metricName));
    exportgraphics(fig,fullfile(outDir,sprintf('Figure_%s.png',metricName)),'Resolution',200);
    savefig(fig,fullfile(outDir,sprintf('Figure_%s.fig',metricName)));
end
fprintf('\nDone. Outputs saved to: %s\n',outDir);
fprintf('Primary comparison: main_common_test_results.csv\n');
fprintf('Repeat with trainingSeeds=[11 22 33] (or more) for stronger repeatability evidence.\n');
fprintf('This script was not executed in MATLAB in this environment; run and check for version-specific issues.\n');

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
    dc=(2*rand(1,3)-1).*dcFrac.*A; h5phase=2*pi*rand(1,3); h7phase=2*pi*rand(1,3); wave=zeros(3,N);
    for pp=1:3
        fundamental=A(pp)*sin(2*pi*f*t+phaseRad(pp));
        fifth=A(pp)*h5(pp)*sin(2*pi*5*f*t+h5phase(pp));
        seventh=A(pp)*h7(pp)*sin(2*pi*7*f*t+h7phase(pp));
        ac=fundamental+fifth+seventh; noiseRMS=rms(ac)/(10^(snrDb/20));
        wave(pp,:)=ac+dc(pp)+noiseRMS*randn(1,N);
    end
    X(ii,:)=single(reshape(wave.',1,[])); Y(ii,:)=[A phaseDeg f];
    meta.SNR_dB(ii)=snrDb; meta.H5MaxFraction(ii)=max(h5); meta.H7MaxFraction(ii)=max(h7);
    meta.DCMaxFraction(ii)=max(dcFrac); meta.FrequencyDeviation_Hz(ii)=f-50; meta.SevereFlag(ii)=double(isSevere);
end
end

function metrics=scoreOutputs(Ytrue,Ypred,severity,M,fs,testSet,seed)
names=["Amplitude A";"Amplitude B";"Amplitude C";"Phase A";"Phase B";"Phase C";"Frequency"];
units=["V peak";"V peak";"V peak";"deg";"deg";"deg";"Hz"];
groups=["Overall";"Moderate";"Severe"]; rows=cell(0,9);
for gi=1:numel(groups)
    if groups(gi)=="Overall", mask=true(size(severity)); else, mask=severity==groups(gi); end
    if ~any(mask), continue; end
    for j=1:7
        e=Ypred(mask,j)-Ytrue(mask,j); if j>=4 && j<=6, e=mod(e+180,360)-180; end
        rows(end+1,:)={fs,M,string(testSet),groups(gi),names(j),units(j),mean(abs(e)),sqrt(mean(e.^2)),seed}; %#ok<AGROW>
    end
end
metrics=cell2table(rows,'VariableNames',{'SamplingFrequency_Hz','DatasetSize','TestSet','Severity','Output','Unit','MAE','RMSE','TrainingSeed'});
end
