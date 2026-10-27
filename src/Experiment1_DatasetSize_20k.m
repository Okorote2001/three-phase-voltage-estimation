%% Experiment 1: Effect of dataset size on 3-phase voltage parameter estimation (including 20,000 examples)
% Fixed: fs=10 kHz, observation window=20 ms, 200 samples/phase, 600 inputs.
% Outputs: [Aamp Bamp Camp Aphase Bphase Cphase frequency].
% Requires Deep Learning Toolbox. All amplitudes/phase targets are peak V/degrees.
clear; clc; close all;
rng(42,'twister');

%% Fixed settings
fs = 10000;
Tobs = 0.020;
t = (0:1/fs:Tobs-1/fs);
N = numel(t);                         % 200 samples per phase
numFeatures = 3*N;                    % 600 input features
datasetSizes = [1000 2000 5000 10000 20000];
VnomRMS = 220;                        % provisional phase-neutral nominal
VnomPeak = sqrt(2)*VnomRMS;           % ~311.13 V peak
phaseNominal = [0 -120 120];

% Signal/label variation
ampMin = 0.90; ampMax = 1.10;         % each phase: 90-110% nominal peak
fMin = 49.5; fMax = 50.5;             % shared frequency per 3-phase case
phaseDeviationMax = 5;                % degrees per phase around nominal

% Disturbances: mixed 80% moderate / 20% severe (experimental assumptions)
severeFraction = 0.20;
h5ModMax = 0.05; h7ModMax = 0.05;     % 0-5% fundamental amplitude
h5SevMin = 0.05; h5SevMax = 0.10;     % 5-10%
h7SevMin = 0.05; h7SevMax = 0.10;
snrMod = [30 40]; snrSev = [20 30];   % dB
dcModMax = 0.02; dcSevMax = 0.05;     % DC magnitude up to 2% / 5% of peak

% Neural network/training settings (fixed architecture across dataset sizes)
bottleneckUnits = 36;                 % Linear subspace filter to reject noise/harmonics
hidden1 = 128;
hidden2 = 64;
miniBatchSize = 64;
initialLearnRate = 3e-3;
l2Reg = 5e-3;                         % L2 regularization prevents noise overfitting

outDir = 'Experiment1_DatasetSize_Results';
if ~exist(outDir,'dir'), mkdir(outDir); end
fprintf('fs = %g Hz | window = %.1f ms | N = %d per phase | inputs = %d\n', ...
    fs,Tobs*1000,N,numFeatures);

%% Generate the 20,000-example pool once
[XpoolRaw,Ypool,severityPool,metaPool] = makeData(max(datasetSizes),t,VnomPeak, ...
    ampMin,ampMax,fMin,fMax,phaseNominal,phaseDeviationMax, ...
    severeFraction,h5ModMax,h7ModMax,h5SevMin,h5SevMax,h7SevMin,h7SevMax, ...
    snrMod,snrSev,dcModMax,dcSevMax);

% Independent common test set: same cases used to compare every dataset-size model.
% This set is additional to each configuration's internal 70/15/15 split.
rng(9001,'twister');
nCommonTest = 1500;
[XcommonRaw,Ycommon,severityCommon,~] = makeData(nCommonTest,t,VnomPeak, ...
    ampMin,ampMax,fMin,fMax,phaseNominal,phaseDeviationMax, ...
    severeFraction,h5ModMax,h7ModMax,h5SevMin,h5SevMax,h7SevMin,h7SevMax, ...
    snrMod,snrSev,dcModMax,dcSevMax);

% Per-example per-phase AC coupling (removes DC offset per window without leakage)
% followed by fixed nominal peak voltage scaling.
Xpool   = preprocessWaveforms(XpoolRaw, N, VnomPeak);
Xcommon = preprocessWaveforms(XcommonRaw, N, VnomPeak);

% Define high-precision architecture:
% 1. Linear bottleneck (36 units) projects 600 noisy samples onto low-dimensional
%    fundamental & harmonic-rejecting subspace before nonlinearities.
% 2. tanh + ReLU hidden layers accurately capture both quadratic amplitude (I^2+Q^2)
%    and linear phase/frequency shifts without half-wave clipping distortion.
layers = [
    featureInputLayer(numFeatures,'Normalization','none','Name','input')
    fullyConnectedLayer(bottleneckUnits,'Name','fc_subspace')
    fullyConnectedLayer(hidden1,'Name','fc1')
    tanhLayer('Name','tanh1')
    fullyConnectedLayer(hidden2,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(7,'Name','output')
    regressionLayer('Name','regression')
];

allMetrics = table();
trainSummary = table();
commonPredictions = table();

%% Train one model per dataset size
for k = 1:numel(datasetSizes)
    M = datasetSizes(k);
    fprintf('\n--- Dataset size: %d ---\n',M);

    % Deterministic random split, exactly 70/15/15 (integer counts).
    rng(1000+M,'twister');
    order = randperm(M);
    nTrain = floor(0.70*M);
    nVal = floor(0.15*M);
    nTest = M-nTrain-nVal;
    iTrain = order(1:nTrain);
    iVal = order(nTrain+1:nTrain+nVal);
    iTest = order(nTrain+nVal+1:end);

    XTrain = Xpool(iTrain,:); XVal = Xpool(iVal,:); XTest = Xpool(iTest,:);
    YTrainRaw = Ypool(iTrain,:); YValRaw = Ypool(iVal,:); YTestRaw = Ypool(iTest,:);

    % Standardize outputs using training-set statistics only.
    yMean = mean(YTrainRaw,1);
    yStd = std(YTrainRaw,0,1);
    yStd(yStd < 1e-12) = 1;
    YTrain = (YTrainRaw-yMean)./yStd;
    YVal = (YValRaw-yMean)./yStd;

    % Ensure smaller datasets receive sufficient gradient steps for quadratic
    % amplitude convergence before piecewise LR decay finishes.
    maxEpochs = max(250, round(450 * (2000 / max(M, 1000))));
    learnRateDropPeriod = round(maxEpochs / 3);
    valFreq = max(5, floor(nTrain / miniBatchSize));

    opts = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'InitialLearnRate',initialLearnRate, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',learnRateDropPeriod, ...
        'LearnRateDropFactor',0.25, ...
        'L2Regularization',l2Reg, ...
        'Shuffle','every-epoch', ...
        'ValidationData',{XVal,YVal}, ...
        'ValidationFrequency',valFreq, ...
        'ValidationPatience',40, ...
        'OutputNetwork','best-validation-loss', ...
        'Verbose',false,'Plots','none','ExecutionEnvironment','auto');

    timerVal = tic;
    net = trainNetwork(XTrain,YTrain,layers,opts);
    trainSeconds = toc(timerVal);

    % Internal held-out test subset
    YhatInternal = double(predict(net,XTest,'MiniBatchSize',miniBatchSize)).*yStd+yMean;
    metricsInternal = scoreOutputs(YTestRaw,YhatInternal,severityPool(iTest),M,"InternalTest");
    allMetrics = [allMetrics; metricsInternal]; %#ok<AGROW>

    % Common independent test set: primary comparison across dataset sizes
    YhatCommon = double(predict(net,Xcommon,'MiniBatchSize',miniBatchSize)).*yStd+yMean;
    metricsCommon = scoreOutputs(Ycommon,YhatCommon,severityCommon,M,"CommonTest");
    allMetrics = [allMetrics; metricsCommon]; %#ok<AGROW>

    trainSummary = [trainSummary; table(M,nTrain,nVal,nTest,trainSeconds, ...
        'VariableNames',{'DatasetSize','NumTraining','NumValidation','NumInternalTest','TrainingSeconds'})]; %#ok<AGROW>

    % Save per-example common-test predictions for further analysis.
    p = table((1:nCommonTest)',severityCommon, ...
        Ycommon(:,1),YhatCommon(:,1),Ycommon(:,2),YhatCommon(:,2), ...
        Ycommon(:,3),YhatCommon(:,3),Ycommon(:,4),YhatCommon(:,4), ...
        Ycommon(:,5),YhatCommon(:,5),Ycommon(:,6),YhatCommon(:,6), ...
        Ycommon(:,7),YhatCommon(:,7), ...
        'VariableNames',{'Case','Severity','TrueAmpA_Vpeak','PredAmpA_Vpeak', ...
        'TrueAmpB_Vpeak','PredAmpB_Vpeak','TrueAmpC_Vpeak','PredAmpC_Vpeak', ...
        'TruePhaseA_deg','PredPhaseA_deg','TruePhaseB_deg','PredPhaseB_deg', ...
        'TruePhaseC_deg','PredPhaseC_deg','TrueFrequency_Hz','PredFrequency_Hz'});
    p.DatasetSize = repmat(M,nCommonTest,1);
    commonPredictions = [commonPredictions; p]; %#ok<AGROW>

    save(fullfile(outDir,sprintf('model_%d_examples.mat',M)), ...
        'net','yMean','yStd','VnomPeak','fs','Tobs','M','t');

    % Common-test true-vs-predicted plot
    labels = {'Amplitude A (V peak)','Amplitude B (V peak)','Amplitude C (V peak)', ...
        'Phase A (deg)','Phase B (deg)','Phase C (deg)','Frequency (Hz)'};
    fig = figure('Visible','off','Color','w','Position',[100 100 1200 750]);
    tiledlayout(3,3,'Padding','compact','TileSpacing','compact');
    for j = 1:7
        nexttile;
        if j>=4 && j<=6
            e = phaseError(YhatCommon(:,j),Ycommon(:,j));
            scatter(Ycommon(:,j),Ycommon(:,j)+e,9,'filled');
        else
            scatter(Ycommon(:,j),YhatCommon(:,j),9,'filled');
        end
        hold on;
        vv = [Ycommon(:,j);YhatCommon(:,j)];
        lo=min(vv); hi=max(vv); if hi<=lo, hi=lo+1; end
        plot([lo hi],[lo hi],'k--','LineWidth',1);
        xlabel(['True ' labels{j}]); ylabel(['Predicted ' labels{j}]);
        title(labels{j}); grid on;
    end
    nexttile; axis off;
    text(0,0.8,sprintf('Dataset size: %d',M),'FontSize',12);
    text(0,0.6,sprintf('Training time: %.1f s',trainSeconds),'FontSize',11);
    exportgraphics(fig,fullfile(outDir,sprintf('true_vs_predicted_%d.png',M)),'Resolution',180);
    close(fig);

    fprintf('Split: %d / %d / %d; training time %.1f s\n',nTrain,nVal,nTest,trainSeconds);
    disp(metricsCommon(metricsCommon.Severity=="Overall",{'Output','MAE','RMSE'}));
end

%% Save tables and experiment settings
writetable(allMetrics,fullfile(outDir,'experiment1_error_metrics.csv'));
writetable(trainSummary,fullfile(outDir,'experiment1_training_summary.csv'));
writetable(commonPredictions,fullfile(outDir,'experiment1_common_test_predictions.csv'));

labelNames = {'AmpA_Vpeak','AmpB_Vpeak','AmpC_Vpeak','PhaseA_deg', ...
    'PhaseB_deg','PhaseC_deg','Frequency_Hz'};
labelTable = array2table(Ypool,'VariableNames',labelNames);
labelTable.Severity = severityPool;
labelTable = [labelTable metaPool];
writetable(labelTable,fullfile(outDir,'master_dataset_labels.csv'));

settings = struct('fs_Hz',fs,'observationWindow_s',Tobs,'samplesPerPhase',N, ...
    'inputFeatures',numFeatures,'outputCount',7,'datasetSizes',datasetSizes, ...
    'split',[0.70 0.15 0.15],'nominalPhaseNeutralRMS_V',VnomRMS, ...
    'nominalPeak_V',VnomPeak,'frequencyRange_Hz',[fMin fMax], ...
    'amplitudeFraction',[ampMin ampMax],'phaseDeviationMax_deg',phaseDeviationMax, ...
    'severeFraction',severeFraction,'moderateHarmonicMaxFraction',[h5ModMax h7ModMax], ...
    'severeHarmonicFractionRange',[h5SevMin h5SevMax], ...
    'moderateSNR_dB',snrMod,'severeSNR_dB',snrSev, ...
    'moderateDCFractionMax',dcModMax,'severeDCFractionMax',dcSevMax, ...
    'bottleneckUnits',bottleneckUnits,'hiddenUnits',[hidden1 hidden2], ...
    'miniBatchSize',miniBatchSize,'initialLearnRate',initialLearnRate, ...
    'l2Regularization',l2Reg,'randomSeedMaster',42);
save(fullfile(outDir,'experiment1_complete_results.mat'), ...
    'allMetrics','trainSummary','commonPredictions','settings');

fprintf('\nExperiment 1 finished. Results are in: %s\n',outDir);
fprintf('Use CommonTest + Overall rows for the main comparison across dataset sizes.\n');

%% Local functions
function Xnorm = preprocessWaveforms(Xraw, N, VnomPeak)
    % Subtract per-phase window mean (removes random DC offset per example)
    % and normalize by nominal peak voltage.
    va = Xraw(:, 1:N);
    vb = Xraw(:, N+1:2*N);
    vc = Xraw(:, 2*N+1:3*N);
    va = va - mean(va, 2);
    vb = vb - mean(vb, 2);
    vc = vc - mean(vc, 2);
    Xnorm = [va, vb, vc] / single(VnomPeak);
end

function [X,Y,severity,meta] = makeData(n,t,Vpk,ampMin,ampMax,fMin,fMax, ...
    phaseNominal,phaseDevMax,severeFraction,h5ModMax,h7ModMax, ...
    h5SevMin,h5SevMax,h7SevMin,h7SevMax,snrMod,snrSev,dcModMax,dcSevMax)
    N = numel(t);
    X = zeros(n,3*N,'single');
    Y = zeros(n,7);
    severity = strings(n,1);
    meta = table('Size',[n 5], ...
        'VariableTypes',{'double','double','double','double','double'}, ...
        'VariableNames',{'SNR_dB','H5MaxFraction','H7MaxFraction','DCMaxFraction','FrequencyDeviation_Hz'});
    isSevere = rand(n,1)<severeFraction;
    for ii=1:n
        A = Vpk*(ampMin+(ampMax-ampMin)*rand(1,3));
        f = fMin+(fMax-fMin)*rand();  % shared frequency for phases A, B, C
        phaseDeg = phaseNominal+(2*rand(1,3)-1)*phaseDevMax;
        phaseRad = deg2rad(phaseDeg);
        if isSevere(ii)
            h5 = h5SevMin+(h5SevMax-h5SevMin)*rand(1,3);
            h7 = h7SevMin+(h7SevMax-h7SevMin)*rand(1,3);
            snrDb = snrSev(1)+(snrSev(2)-snrSev(1))*rand();
            dcFrac = dcSevMax*rand(1,3);
            severity(ii)="Severe";
        else
            h5 = h5ModMax*rand(1,3);
            h7 = h7ModMax*rand(1,3);
            snrDb = snrMod(1)+(snrMod(2)-snrMod(1))*rand();
            dcFrac = dcModMax*rand(1,3);
            severity(ii)="Moderate";
        end
        dc = (2*rand(1,3)-1).*dcFrac.*A;
        h5phase = 2*pi*rand(1,3);
        h7phase = 2*pi*rand(1,3);
        wave = zeros(3,N);
        for pp=1:3
            fundamental = A(pp)*sin(2*pi*f*t+phaseRad(pp));
            fifth = A(pp)*h5(pp)*sin(2*pi*5*f*t+h5phase(pp));
            seventh = A(pp)*h7(pp)*sin(2*pi*7*f*t+h7phase(pp));
            ac = fundamental+fifth+seventh;
            noiseRMS = rms(ac)/(10^(snrDb/20));
            wave(pp,:) = ac+dc(pp)+noiseRMS*randn(1,N);
        end
        % FIXED: Transpose wave.' so row 1 (Phase A), row 2 (Phase B), and
        % row 3 (Phase C) are stored contiguously as [Va(1:N), Vb(1:N), Vc(1:N)].
        X(ii,:) = single(reshape(wave.',1,[]));
        Y(ii,:) = [A phaseDeg f];
        meta.SNR_dB(ii)=snrDb;
        meta.H5MaxFraction(ii)=max(h5);
        meta.H7MaxFraction(ii)=max(h7);
        meta.DCMaxFraction(ii)=max(dcFrac);
        meta.FrequencyDeviation_Hz(ii)=f-50;
    end
end

function metrics = scoreOutputs(Ytrue,Ypred,severity,datasetSize,testSet)
    names = ["Amplitude A (V peak)";"Amplitude B (V peak)"; ...
        "Amplitude C (V peak)";"Phase A (deg)";"Phase B (deg)"; ...
        "Phase C (deg)";"Frequency (Hz)"];
    units = ["V peak";"V peak";"V peak";"deg";"deg";"deg";"Hz"];
    groups = ["Overall";"Moderate";"Severe"];
    rows = cell(0,7);
    for gg=1:numel(groups)
        if groups(gg)=="Overall"
            mask=true(size(severity));
        else
            mask=severity==groups(gg);
        end
        if ~any(mask), continue; end
        for jj=1:7
            err=Ypred(mask,jj)-Ytrue(mask,jj);
            if jj>=4 && jj<=6
                err=phaseError(Ypred(mask,jj),Ytrue(mask,jj));
            end
            rows(end+1,:)={datasetSize,string(testSet),groups(gg),names(jj),units(jj), ...
                mean(abs(err)),sqrt(mean(err.^2))}; %#ok<AGROW>
        end
    end
    metrics=cell2table(rows,'VariableNames', ...
        {'DatasetSize','TestSet','Severity','Output','Unit','MAE','RMSE'});
end

function e = phaseError(predDeg,trueDeg)
    e=mod(predDeg-trueDeg+180,360)-180;
end