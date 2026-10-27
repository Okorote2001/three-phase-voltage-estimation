%% Experiment 2: Sampling frequency x dataset size for 3-phase voltage estimation
% Gemini-style preprocessing/network, with a corrected contiguous phase input layout.
% Factors:
%   Sampling frequency: 1, 2, 5, 10 kHz
%   Dataset size:       2k, 5k, 10k, 15k, 20k examples
% Fixed observation window: 20 ms
% Fixed training protocol: maximum 300 epochs, LR drop every 100 epochs,
% validation-based best-network selection and early stopping.
% Outputs: [AmpA AmpB AmpC PhaseA PhaseB PhaseC Frequency]
%
% Requires MATLAB Deep Learning Toolbox.
% Amplitudes are peak volts; phases are degrees; frequency is Hz.
% IMPORTANT: The 1 kHz case is close to the minimum useful sampling rate for
% representing the 7th harmonic (350 Hz). Results should be interpreted as an
% experimental comparison, not a claim that 1 kHz is universally sufficient.

clear; clc; close all;
rng(42,'twister');

%% Experimental factors
fs = 10000;                                  % fixed sampling frequency (Hz)
observationWindows = [0.002 0.005 0.010 0.015 0.020]; % 2, 5, 10, 15, 20 ms
datasetSizes = [2000 5000 10000 15000 20000];
VnomRMS = 220;
VnomPeak = sqrt(2)*VnomRMS;
phaseNominal = [0 -120 120];

% Target/condition ranges (same assumptions as Gemini-style Experiment 1)
ampMin = 0.90; ampMax = 1.10;
fMin = 49.5; fMax = 50.5;
phaseDeviationMax = 5;
severeFraction = 0.20;
h5ModMax = 0.05; h7ModMax = 0.05;
h5SevMin = 0.05; h5SevMax = 0.10;
h7SevMin = 0.05; h7SevMax = 0.10;
snrMod = [30 40]; snrSev = [20 30];
dcModMax = 0.02; dcSevMax = 0.05;

% Gemini-style network/training settings
bottleneckUnits = 36;
hidden1 = 128;
hidden2 = 64;
miniBatchSize = 64;
initialLearnRate = 3e-3;
l2Reg = 5e-3;

nCommonTest = 1500; % separate fixed evaluation set per sampling frequency
masterSeed = 42000;
commonSeed = 900100;
outDir = 'Experiment3_ObservationWindow_x_DatasetSize_Results';
if ~exist(outDir,'dir'), mkdir(outDir); end

fprintf('Experiment 3: observation window x dataset size\n');
fprintf('Fixed sampling frequency = %g Hz (%.1f kHz)\n',fs,fs/1000);
fprintf('Observation windows (ms) = %s\n',mat2str(observationWindows*1000));
fprintf('Dataset sizes = %s\n\n',mat2str(datasetSizes));

allMetrics = table();
trainSummary = table();
commonPredictions = table();

%% Run each observation window; compare all dataset sizes at fixed fs
for wi = 1:numel(observationWindows)
    Tobs = observationWindows(wi);
    t = (0:1/fs:Tobs-1/fs);
    N = numel(t);
    numFeatures = 3*N;

    fprintf('\n============================================================\n');
    fprintf('Observation window: %.1f ms | fs: %g Hz | N/phase: %d | inputs: %d\n', ...
        Tobs*1000,fs,N,numFeatures);
    fprintf('============================================================\n');

    % Generate one 20k master pool and a separate common test set for this window.
    % Case-level seeds preserve underlying labels/disturbance settings across windows;
    % waveform samples/noise differ because the observation duration differs.
    [XpoolRaw,Ypool,severityPool,metaPool] = makeData(max(datasetSizes),t, ...
        VnomPeak,ampMin,ampMax,fMin,fMax,phaseNominal,phaseDeviationMax, ...
        severeFraction,h5ModMax,h7ModMax,h5SevMin,h5SevMax,h7SevMin,h7SevMax, ...
        snrMod,snrSev,dcModMax,dcSevMax,masterSeed);

    [XcommonRaw,Ycommon,severityCommon,~] = makeData(nCommonTest,t, ...
        VnomPeak,ampMin,ampMax,fMin,fMax,phaseNominal,phaseDeviationMax, ...
        severeFraction,h5ModMax,h7ModMax,h5SevMin,h5SevMax,h7SevMin,h7SevMax, ...
        snrMod,snrSev,dcModMax,dcSevMax,commonSeed);

    Xpool = preprocessWaveforms(XpoolRaw,N,VnomPeak);
    Xcommon = preprocessWaveforms(XcommonRaw,N,VnomPeak);

    for k = 1:numel(datasetSizes)
        M = datasetSizes(k);
        fprintf('\n--- window = %.1f ms | dataset size = %d ---\n',Tobs*1000,M);

        % Each dataset size uses the first M cases from the same master pool.
        % Each size is split into 70/15/15 using a reproducible random permutation.
        % Because each size is split independently, training subsets are not
        % strictly nested; the fixed common test set is the primary comparison.
        rng(100000 + round(Tobs*1e6) + M,'twister');
        order = randperm(M);
        nTrain = floor(0.70*M);
        nVal = floor(0.15*M);
        nTest = M-nTrain-nVal;
        iTrain = order(1:nTrain);
        iVal = order(nTrain+1:nTrain+nVal);
        iTest = order(nTrain+nVal+1:end);

        XTrain = Xpool(iTrain,:); XVal = Xpool(iVal,:); XTest = Xpool(iTest,:);
        YTrainRaw = Ypool(iTrain,:); YValRaw = Ypool(iVal,:); YTestRaw = Ypool(iTest,:);

        % Standardize labels using training data only.
        yMean = mean(YTrainRaw,1);
        yStd = std(YTrainRaw,0,1);
        yStd(yStd < 1e-12) = 1;
        YTrain = (YTrainRaw-yMean)./yStd;
        YVal = (YValRaw-yMean)./yStd;

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

        % FIX: use the same maximum epoch budget and LR schedule for every
        % dataset size and sampling frequency. Validation selects the best
        % checkpoint and early stopping can terminate training sooner.
        % This makes the training protocol more comparable across configurations.
        maxEpochs = 300;
        learnRateDropPeriod = 100;
        valFreq = max(5,floor(nTrain/miniBatchSize));

        % Reproducible, configuration-specific network initialization/shuffling.
        rng(200000 + round(Tobs*1e6) + M,'twister');

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
            'Verbose',false,'Plots','training-progress','ExecutionEnvironment','auto');

        ticTrain = tic;
        net = trainNetwork(XTrain,YTrain,layers,opts);
        trainSeconds = toc(ticTrain);

        % Internal test and common test; common test is primary for comparison.
        YhatInternal = double(predict(net,XTest,'MiniBatchSize',miniBatchSize)).*yStd+yMean;
        metricsInternal = scoreOutputs(YTestRaw,YhatInternal,severityPool(iTest), ...
            M,fs,"InternalTest");
        metricsInternal.Window_ms = repmat(Tobs*1000,height(metricsInternal),1);
        allMetrics = [allMetrics;metricsInternal]; %#ok<AGROW>

        YhatCommon = double(predict(net,Xcommon,'MiniBatchSize',miniBatchSize)).*yStd+yMean;
        metricsCommon = scoreOutputs(Ycommon,YhatCommon,severityCommon,M,fs,"CommonTest");
        metricsCommon.Window_ms = repmat(Tobs*1000,height(metricsCommon),1);
        allMetrics = [allMetrics;metricsCommon]; %#ok<AGROW>

        trainSummary = [trainSummary; table(fs,Tobs*1000,Tobs,M,N,numFeatures,nTrain,nVal,nTest, ...
            trainSeconds,maxEpochs,'VariableNames',{'SamplingFrequency_Hz','Window_ms','Window_s', ...
            'DatasetSize','SamplesPerPhase','InputFeatures','NumTraining','NumValidation', ...
            'NumInternalTest','TrainingSeconds','MaxEpochs'})]; %#ok<AGROW>

        ptab = table(repmat(fs,nCommonTest,1),repmat(Tobs*1000,nCommonTest,1),repmat(M,nCommonTest,1), ...
            (1:nCommonTest)',severityCommon, ...
            Ycommon(:,1),YhatCommon(:,1),Ycommon(:,2),YhatCommon(:,2), ...
            Ycommon(:,3),YhatCommon(:,3),Ycommon(:,4),YhatCommon(:,4), ...
            Ycommon(:,5),YhatCommon(:,5),Ycommon(:,6),YhatCommon(:,6), ...
            Ycommon(:,7),YhatCommon(:,7), ...
            'VariableNames',{'SamplingFrequency_Hz','Window_ms','DatasetSize','Case','Severity', ...
            'TrueAmpA_Vpeak','PredAmpA_Vpeak','TrueAmpB_Vpeak','PredAmpB_Vpeak', ...
            'TrueAmpC_Vpeak','PredAmpC_Vpeak','TruePhaseA_deg','PredPhaseA_deg', ...
            'TruePhaseB_deg','PredPhaseB_deg','TruePhaseC_deg','PredPhaseC_deg', ...
            'TrueFrequency_Hz','PredFrequency_Hz'});
        commonPredictions = [commonPredictions;ptab]; %#ok<AGROW>

        safeWindow = sprintf('%gms',Tobs*1000);
        save(fullfile(outDir,sprintf('model_%s_%d_examples.mat',safeWindow,M)), ...
            'net','yMean','yStd','VnomPeak','fs','Tobs','M','t');

        fprintf('Split %d / %d / %d | %.1f s training\n', ...
            nTrain,nVal,nTest,trainSeconds);
        disp(metricsCommon(metricsCommon.Severity=="Overall", ...
            {'Output','Unit','MAE','RMSE'}));
    end

    % Save labels for the pool at this sampling rate (targets are the same by design).
    labelNames = {'AmpA_Vpeak','AmpB_Vpeak','AmpC_Vpeak','PhaseA_deg', ...
        'PhaseB_deg','PhaseC_deg','Frequency_Hz'};
    labels = array2table(Ypool,'VariableNames',labelNames);
    labels.Severity = severityPool;
    labels = [table((1:height(labels))','VariableNames',{'Case'}) labels metaPool];
    writetable(labels,fullfile(outDir,sprintf('master_labels_%gms.csv',Tobs*1000)));
end

%% Save results
writetable(allMetrics,fullfile(outDir,'experiment3_all_error_metrics.csv'));
writetable(trainSummary,fullfile(outDir,'experiment3_training_summary.csv'));
writetable(commonPredictions,fullfile(outDir,'experiment3_common_test_predictions.csv'));

% Main comparison table: common test, overall disturbance severity.
mainMask = allMetrics.TestSet=="CommonTest" & allMetrics.Severity=="Overall";
mainResults = allMetrics(mainMask,:);
writetable(mainResults,fullfile(outDir,'experiment3_main_common_test_results.csv'));

settings = struct('samplingFrequency_Hz',fs,'observationWindows_s',observationWindows, ...
    'datasetSizes',datasetSizes,'observationWindow_s',Tobs, ...
    'split',[0.70 0.15 0.15],'commonTestSize',nCommonTest, ...
    'nominalPhaseNeutralRMS_V',VnomRMS,'nominalPeak_V',VnomPeak, ...
    'frequencyRange_Hz',[fMin fMax],'amplitudeFraction',[ampMin ampMax], ...
    'phaseDeviationMax_deg',phaseDeviationMax,'severeFraction',severeFraction, ...
    'moderateHarmonicMaxFraction',[h5ModMax h7ModMax], ...
    'severeHarmonicFractionRange',[h5SevMin h5SevMax], ...
    'moderateSNR_dB',snrMod,'severeSNR_dB',snrSev, ...
    'moderateDCFractionMax',dcModMax,'severeDCFractionMax',dcSevMax, ...
    'bottleneckUnits',bottleneckUnits,'hiddenUnits',[hidden1 hidden2], ...
    'miniBatchSize',miniBatchSize,'initialLearnRate',initialLearnRate, ...
    'l2Regularization',l2Reg,'maxEpochs',300,'learnRateDropPeriod',100, ...
    'validationPatience',40,'masterSeed',masterSeed,'commonSeed',commonSeed);
save(fullfile(outDir,'experiment3_complete_results.mat'), ...
    'allMetrics','trainSummary','commonPredictions','mainResults','settings');

fprintf('\nExperiment 2 finished. Results saved in: %s\n',outDir);
fprintf('Use experiment3_main_common_test_results.csv for primary comparisons.\n');

%% Publication-style summary figures
% Figure 1: MAE trends against observation window, one panel per parameter.
outputs = ["Amplitude A","Amplitude B","Amplitude C", ...
           "Phase A","Phase B","Phase C","Frequency"];
units = ["V peak","V peak","V peak","deg","deg","deg","Hz"];
fig1 = figure('Name','Experiment 3 - MAE vs Observation Window','Color','w', ...
    'Position',[80 80 1400 800]);
tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
for oi=1:numel(outputs)
    nexttile; hold on; grid on; box on;
    for mi=1:numel(datasetSizes)
        M=datasetSizes(mi);
        mask=mainResults.DatasetSize==M & mainResults.Output==outputs(oi);
        rows=mainResults(mask,:);
        [~,ord]=sort(rows.Window_ms);
        plot(rows.Window_ms(ord),rows.MAE(ord),'-o','LineWidth',1.5, ...
            'MarkerSize',5,'DisplayName',sprintf('%g examples',M));
    end
    title(outputs(oi)); xlabel('Observation window (ms)');
    ylabel(sprintf('MAE (%s)',units(oi)));
    legend('Location','best');
end
sgtitle(sprintf('Common-test MAE versus Observation Window (fs = %g kHz)',fs/1000));
exportgraphics(fig1,fullfile(outDir,'Figure1_MAE_vs_ObservationWindow.png'),'Resolution',200);
savefig(fig1,fullfile(outDir,'Figure1_MAE_vs_ObservationWindow.fig'));

% Figure 2: RMSE trends against dataset size.
fig2 = figure('Name','Experiment 3 - RMSE vs Observation Window','Color','w', ...
    'Position',[80 80 1400 800]);
tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
for oi=1:numel(outputs)
    nexttile; hold on; grid on; box on;
    for mi=1:numel(datasetSizes)
        M=datasetSizes(mi);
        mask=mainResults.DatasetSize==M & mainResults.Output==outputs(oi);
        rows=mainResults(mask,:);
        [~,ord]=sort(rows.Window_ms);
        plot(rows.Window_ms(ord),rows.RMSE(ord),'-o','LineWidth',1.5, ...
            'MarkerSize',5,'DisplayName',sprintf('%g examples',M));
    end
    title(outputs(oi)); xlabel('Observation window (ms)');
    ylabel(sprintf('RMSE (%s)',units(oi)));
    legend('Location','best');
end
sgtitle(sprintf('Common-test RMSE versus Observation Window (fs = %g kHz)',fs/1000));
exportgraphics(fig2,fullfile(outDir,'Figure2_RMSE_vs_ObservationWindow.png'),'Resolution',200);
savefig(fig2,fullfile(outDir,'Figure2_RMSE_vs_ObservationWindow.fig'));

% Figure 3: True-versus-predicted diagnostic for the largest dataset and longest window.
% This uses the saved common test cases and shows all seven outputs.
chosenM=max(datasetSizes); chosenWindow_ms=max(observationWindows)*1000;
sel=commonPredictions.Window_ms==chosenWindow_ms & ...
    commonPredictions.DatasetSize==chosenM;
if any(sel)
    P=commonPredictions(sel,:);
    trueVals={P.TrueAmpA_Vpeak,P.TrueAmpB_Vpeak,P.TrueAmpC_Vpeak, ...
        P.TruePhaseA_deg,P.TruePhaseB_deg,P.TruePhaseC_deg,P.TrueFrequency_Hz};
    predVals={P.PredAmpA_Vpeak,P.PredAmpB_Vpeak,P.PredAmpC_Vpeak, ...
        P.PredPhaseA_deg,P.PredPhaseB_deg,P.PredPhaseC_deg,P.PredFrequency_Hz};
    fig3=figure('Name','True vs Predicted - 20 ms, 20k examples','Color','w', ...
        'Position',[80 80 1400 750]);
    tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
    for oi=1:7
        nexttile; hold on; grid on; box on;
        scatter(trueVals{oi},predVals{oi},12,'filled','MarkerFaceAlpha',0.45);
        lo=min([trueVals{oi}(:);predVals{oi}(:)]);
        hi=max([trueVals{oi}(:);predVals{oi}(:)]);
        if hi<=lo, hi=lo+1; end
        plot([lo hi],[lo hi],'r--','LineWidth',1.5);
        title(outputs(oi)); xlabel(['True (' char(units(oi)) ')']);
        ylabel(['Predicted (' char(units(oi)) ')']); axis tight;
    end
    nexttile; axis off;
    text(0,0.8,sprintf('Observation window: %g ms',chosenWindow_ms),'FontSize',11);
    text(0,0.6,sprintf('Sampling frequency: %g kHz',fs/1000),'FontSize',11);
    text(0,0.4,sprintf('Training dataset: %g examples',chosenM),'FontSize',11);
    text(0,0.2,sprintf('Common test cases: %d',height(P)),'FontSize',11);
    sgtitle('True versus Predicted Parameters (Common Test Set)');
    exportgraphics(fig3,fullfile(outDir,'Figure3_True_vs_Predicted_20ms_20k.png'),'Resolution',200);
    savefig(fig3,fullfile(outDir,'Figure3_True_vs_Predicted_20ms_20k.fig'));
end

fprintf('Figures saved to %s\n',outDir);

%% Local functions
function Xnorm = preprocessWaveforms(Xraw,N,VnomPeak)
    % Gemini-style per-phase window mean removal (AC coupling), then fixed scaling.
    va = Xraw(:,1:N);
    vb = Xraw(:,N+1:2*N);
    vc = Xraw(:,2*N+1:3*N);
    va = va-mean(va,2);
    vb = vb-mean(vb,2);
    vc = vc-mean(vc,2);
    Xnorm = [va,vb,vc]/single(VnomPeak);
end

function [X,Y,severity,meta] = makeData(n,t,Vpk,ampMin,ampMax,fMin,fMax, ...
    phaseNominal,phaseDevMax,severeFraction,h5ModMax,h7ModMax, ...
    h5SevMin,h5SevMax,h7SevMin,h7SevMax,snrMod,snrSev,dcModMax,dcSevMax,baseSeed)
    N = numel(t);
    X = zeros(n,3*N,'single');
    Y = zeros(n,7);
    severity = strings(n,1);
    meta = table('Size',[n 5], ...
        'VariableTypes',{'double','double','double','double','double'}, ...
        'VariableNames',{'SNR_dB','H5MaxFraction','H7MaxFraction', ...
        'DCMaxFraction','FrequencyDeviation_Hz'});

    for ii=1:n
        % Reset per-case seed so the underlying labels/disturbance parameters
        % are consistent across fs; waveform noise is regenerated at each rate.
        rng(baseSeed+ii,'twister');
        A = Vpk*(ampMin+(ampMax-ampMin)*rand(1,3));
        f = fMin+(fMax-fMin)*rand();
        phaseDeg = phaseNominal+(2*rand(1,3)-1)*phaseDevMax;
        phaseRad = deg2rad(phaseDeg);
        isSevere = rand()<severeFraction;

        if isSevere
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

        % Contiguous phase blocks: [Va(1:N), Vb(1:N), Vc(1:N)].
        X(ii,:) = single(reshape(wave.',1,[]));
        Y(ii,:) = [A phaseDeg f];
        meta.SNR_dB(ii)=snrDb;
        meta.H5MaxFraction(ii)=max(h5);
        meta.H7MaxFraction(ii)=max(h7);
        meta.DCMaxFraction(ii)=max(dcFrac);
        meta.FrequencyDeviation_Hz(ii)=f-50;
    end
end

function metrics = scoreOutputs(Ytrue,Ypred,severity,datasetSize,fs,testSet)
    names = ["Amplitude A";"Amplitude B";"Amplitude C"; ...
        "Phase A";"Phase B";"Phase C";"Frequency"];
    units = ["V peak";"V peak";"V peak";"deg";"deg";"deg";"Hz"];
    groups = ["Overall";"Moderate";"Severe"];
    rows = cell(0,8);
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
                err=mod(err+180,360)-180;
            end
            rows(end+1,:)={fs,datasetSize,string(testSet),groups(gg), ...
                names(jj),units(jj),mean(abs(err)),sqrt(mean(err.^2))}; %#ok<AGROW>
        end
    end
    metrics=cell2table(rows,'VariableNames',{'SamplingFrequency_Hz','DatasetSize', ...
        'TestSet','Severity','Output','Unit','MAE','RMSE'});
end
