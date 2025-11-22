% Copyright © 2022-2030, Tianyuan Li, Department of Electronic Engineering, Tsinghua University. All Rights Reserved.
function runPtychoReconGPU(cfg_in, logFileID, gpuDev)
    % run ptychography reconstruction on gpu
    
    %% (I) set the parameters
    % I.1 set and calculate all the parameters involved in reconstruction process
    oExpParas = paras.cPtychoExpParas;
    oExpParas.init(cfg_in);
    oReconParas = paras.cPtychoReconParas;
    oReconParas.init(oExpParas, cfg_in);
    io.DispAndLog(logFileID, ['The pSize is ', num2str(oReconParas.pSize*1e10, '%.3f'), ' A']);
    io.DispAndLog(logFileID, ['The pSizeF is ', num2str(oReconParas.pSizeF*1e-10, '%.6f'), ' A-1']);
    io.DispAndLog(logFileID, ['The probN is ', num2str(oReconParas.probN)]);
    
    %% (II) perpare for reconstruction
    % II.1 load cbed data
    if oReconParas.isLoadingCBED
        load(oReconParas.savedCBEDFile); % oCBED
        if oCBED.checkParas(oExpParas.diffN, oExpParas.scanN, ...
                oReconParas.updownsamplingF, oReconParas.updownsamplingR, ...
                oReconParas.subAreaScanIDs, ...
                oReconParas.nZeroEdges, ...
                oExpParas.darkReference, oExpParas.gainReference, ...
                oExpParas.ADU, oReconParas.flipMode, ...
                oReconParas.cbedTrans, ...
                oReconParas.cbedMask)
            error('CBED cannot be loaded directly!');
        end
    else
        oCBED = cCBED(oExpParas.diffN, oExpParas.scanN, ...
            'updownsamplingF', oReconParas.updownsamplingF, 'updownsamplingR', oReconParas.updownsamplingR, ...
            'subAreaScanIDs', oReconParas.subAreaScanIDs);
        oCBED.loadAndPreprocess(oReconParas.path_cbeds, ...
            'nZeroEdges', oReconParas.nZeroEdges, ...
            'darkReference', oExpParas.darkReference, 'gainReference', oExpParas.gainReference, ...
            'ADU', oExpParas.ADU, 'flipMode', oReconParas.flipMode);
        oCBED.alignment(oReconParas.cbedTrans, gpuDev);
        oCBED.maskCBED(oReconParas.cbedMask);
        oCBED.updownsampling();
        if oReconParas.isSavingCBED
            save(oReconParas.savedCBEDFile, 'oCBED', '-v7.3');
        end
    end
    % arrange cbeds in the order of beats, load cbeds onto gpu
    oCBEDReconG = cCBEDReconGSimplified(); % ifftshift cbeds in this function
    oCBEDReconG.setBeats(oCBED, oReconParas.beats, oReconParas.isCBEDOnGPU);

    % II.2 init object function (potential or complex)
    oObjG = objects.cObjG;
    if oReconParas.isLoadObj % load from file
        switch oReconParas.objModel
            case 'potential'
                objAngle = io.ReadMRC(oReconParas.path_objToLoad{2});
                objAngle = padarray(objAngle, [oReconParas.nPadForLoad oReconParas.nPadForLoad], mean(objAngle, 'all'), 'both');
                % check the size of the loaded object
                if size(objAngle, 1) ~= oReconParas.objNY || size(objAngle, 2) ~= oReconParas.objNX || size(objAngle, 3) ~= oReconParas.nSlices
                    error('The size of the loaded object does not match the settings!');
                end
                oObjG.initFromData(objAngle, ...
                    oReconParas.pSize, oReconParas.sliThick, oReconParas.oriIndY, oReconParas.oriIndX);
                clear objAngle;
            case 'complex'
                objAbs = io.ReadMRC(oReconParas.path_objToLoad{1});
                objAngle = io.ReadMRC(oReconParas.path_objToLoad{2});
                objAbs = padarray(objAbs, [oReconParas.nPadForLoad oReconParas.nPadForLoad], mean(objAbs, 'all'), 'both');
                objAngle = padarray(objAngle, [oReconParas.nPadForLoad oReconParas.nPadForLoad], mean(objAngle, 'all'), 'both');
                % check the size of the loaded object
                if size(objAbs, 1) ~= oReconParas.objNY || size(objAbs, 2) ~= oReconParas.objNX || size(objAbs, 3) ~= oReconParas.nSlices
                    error('The size of the loaded object abs does not match the settings!');
                end
                if size(objAngle, 1) ~= oReconParas.objNY || size(objAngle, 2) ~= oReconParas.objNX || size(objAngle, 3) ~= oReconParas.nSlices
                    error('The size of the loaded object angle does not match the settings!');
                end
                oObjG.initFromData(objAbs .* exp(1i .* objAngle), ...
                    oReconParas.pSize, oReconParas.sliThick, oReconParas.oriIndY, oReconParas.oriIndX);
                clear objAbs objAngle;
            otherwise
                error('wrong object model');
        end
    else % init from scratch
        switch oReconParas.objModel
            case 'potential'
                oObjG.initZeros(oReconParas.objNY, oReconParas.objNX, oReconParas.nSlices, ...
                    oReconParas.pSize, oReconParas.sliThick, oReconParas.oriIndY, oReconParas.oriIndX, ...
                    'single');
            case 'complex'
                oObjG.initOnes(oReconParas.objNY, oReconParas.objNX, oReconParas.nSlices, ...
                    oReconParas.pSize, oReconParas.sliThick, oReconParas.oriIndY, oReconParas.oriIndX, ...
                    'complex');
            otherwise
                error('wrong object model');
        end
    end
    io.DispAndLog(logFileID, ['The object: ', num2str(oReconParas.objNY), 'pix, pSizeFObj = ', num2str(oObjG.pSizeFObj()*1e-10, '%.6f'), 'A-1']);

    % II.3 init probe function (complex)
    oProbG = probes.cProbeG;
    if oReconParas.isLoadProb % load from file
        probAbs = io.ReadMRC(oReconParas.path_probToLoad{1});
        probAngle = io.ReadMRC(oReconParas.path_probToLoad{2});
        % check the size of the loaded probe
        if size(probAbs, 1) ~= oReconParas.probN || size(probAbs, 2) ~= oReconParas.probN || size(probAbs, 3) ~= oReconParas.nModes || ...
                size(probAngle, 1) ~= oReconParas.probN || size(probAngle, 2) ~= oReconParas.probN || size(probAngle, 3) ~= oReconParas.nModes
            error('The size of the loaded probe does not match the settings!');
        end
        oProbG.initFromData(probAbs .* exp(1i .* probAngle), ...
            oReconParas.pSize, oExpParas.waveLength, oExpParas.defocus, oExpParas.Cs, oExpParas.semiAngle);
        clear probAbs probAngle;
        if oReconParas.isAdjustProbIntensity
            oProbG.normalizeGPU(mean(sum(oCBED.cbed, [1 2]), 'all') / (oReconParas.probN)^2, ... % parsaval's theorem
                oReconParas.slaveModeWeight);
        end
    else % init from scratch
        if oReconParas.isInitUsingVacuumProbe
            vacuumProbe = io.ReadMRC(oReconParas.path_vaccumProbe);
        else
            vacuumProbe = [];
        end
        oProbG.initBase(oReconParas.probN, oReconParas.pSizeF, oExpParas.waveLength, ...
            oExpParas.defocus, oExpParas.Cs, oExpParas.semiAngle, ...
            vacuumProbe, ...
            'upsamplingFFactor', oReconParas.upsamplingProbFFactor, ...
            'betaRaisedCosine', 0.0);
        oProbG.initMultiModes(oReconParas.nModes);
        oProbG.orthogonalizeGPU();
        oProbG.normalizeGPU(mean(sum(oCBED.cbed, [1 2]), 'all') / (oReconParas.probN)^2, ... % parsaval's theorem
            oReconParas.slaveModeWeight);
    end
    io.DispAndLog(logFileID, ['The base probe intensity is ', num2str(sum(abs(oProbG.probG(:, :, 1)).^2, 'all'), '%.4f')]);

    % II.4 init spots
    oSpot = cSpot;
    oSpot.setBeats(oReconParas.subAreaScanCoors, oReconParas.beats);
    io.DispAndLog(logFileID, ['Total number of scanning positions is ', num2str(oReconParas.subAreaScanN), ', nSpotsParallel is ', num2str(oReconParas.nSpotsParallel), ...
        ', nBeats is ', num2str(oReconParas.nBeats)]);
    
    % II.5 core for reconstruction
    oReconMG = recons.cReconProSimplifiedG;
    oReconMG.initForwarding(oReconParas.objModel, oReconParas.probN, oReconParas.nSubSpotsParallelM);
    oReconMG.initBackwardingPartA(oReconParas.gradFMask);
    oReconMG.initBackwardingPartB(oObjG.nY(), oObjG.nX(), oObjG.pSize, oObjG.oriIndY, oObjG.oriIndX, ...
        oReconParas.stepSizeFactor);
    io.DispAndLog(logFileID, ['The core M computes ', num2str(oReconParas.nSubSpotsParallelM), ' spots simultaneously, ', ...
        'nSubBeatsM is ', num2str(oReconParas.nSubBeatsM)]);

    % II.6 core for monitoring log-likelihoods
    if oReconParas.isMonitorLikelihoods
        oReconLG = recons.cReconProSimplifiedG;
        oReconLG.initForwarding(oReconParas.objModel, oReconParas.probN, oReconParas.nSubSpotsParallelM);
        maskFMonitorLikelihoodsG = gpuArray(ifftshift(ifftshift(monitorLikelihoodsMask, 1), 2));
        io.DispAndLog(logFileID, ['nSubSpotsParallelL is ', num2str(oReconParas.nSubSpotsParallelL), ...
            ', the core L computes ', num2str(oReconParas.nSubSpotsParallelL), ' spots simultaneously, ', ...
            'nSubBeatsL is ', num2str(oReconParas.nSubBeatsL)]);
    end

    % II.7 dose estimation [e-/A2]
    oIlluIntensityG = objects.cObjG;
    oIlluIntensityG.initZeros(oReconParas.objNY, oReconParas.objNX, 1, ...
        oReconParas.pSize, oReconParas.sliThick, oReconParas.oriIndY, oReconParas.oriIndX, ...
        'single');
    probIntensityG = oProbG.getProbIntensityGPU();
    probIntensityG = probIntensityG / sum(probIntensityG, 'all'); % normalized to 1.0 (probability amplitude)
    probIntensityG = repmat(probIntensityG, [1, 1, oReconParas.nSubSpotsParallelM]);
    for iBeat = 1:oReconParas.nBeats
        for iSubBeat = 1:oReconParas.nSubBeatsM
            [cys1SubBeat, cxs1SubBeat, ~, ~] = oSpot.getSubBeatScanCoors(iBeat, iSubBeat, oReconParas.nSubSpotsParallelM);
            oIlluIntensityG.fuseAddVolumesGPU(probIntensityG, ...
                cys1SubBeat, cxs1SubBeat, ones(oReconParas.nSubSpotsParallelM, 1, 'single', 'gpuArray'), ...
                'isInterpolation', false);
        end
    end
    illuArea = gather(sum(oIlluIntensityG.dataG > 0.25*max(oIlluIntensityG.dataG, [], 'all'), 'all') * oReconParas.pSize^2); % [m^2]
    object_shrinkage_maskG = oIlluIntensityG.dataG < 0.25*max(oIlluIntensityG.dataG, [], 'all'); % (objNY, objNX, 1) bool
    dose1 = oCBED.totElectrons / (illuArea * 1e20); % e-/A2
    dose2 = oCBED.totElectrons / (oReconParas.subArea^2 * 1e20); % e-/A2
    io.DispAndLog(logFileID, ['The dose is ', num2str(dose1, '%.2f'), ' e-/A2 (probIntensity) or ', ...
        num2str(dose2, '%.2f'), ' e-/A2 (scanArea)']);

    % II.8 save initial data
    if oReconParas.saveInit
        path_probAbs = fullfile(oReconParas.path_saveOutputs, 'probAbs000.mrcs');
        path_probAngle = fullfile(oReconParas.path_saveOutputs, 'probAngle000.mrcs');
        oProbG.save(path_probAbs, path_probAngle);
        switch oReconParas.objModel
            case 'potential'
                path_objPot = fullfile(oReconParas.path_saveOutputs, 'objAngle000.mrcs');
                oObjG.savePot(path_objPot);
            case 'complex'
                path_objAbs = fullfile(oReconParas.path_saveOutputs, 'objAbs000.mrcs');
                path_objAngle = fullfile(oReconParas.path_saveOutputs, 'objAngle000.mrcs');
                oObjG.save(path_objAbs, path_objAngle);
            otherwise
                error('Wrong object model!');
        end
        path_coors = fullfile(oReconParas.path_saveOutputs, 'spot000.mat');
        oSpot.saveCoors(path_coors, oExpParas.scanN, oReconParas.subAreaScanIDs);
    end

    %% (III) reconstruction
    for iRound = 1:oReconParas.nRounds

        if oReconParas.isTicToc
            % timer 1 start
            wait(gpuDev);
            timer1 = tic;
        end
        
        updateObj = ismember(iRound, oReconParas.iRoundsUpdateObj);
        updateProb = ismember(iRound, oReconParas.iRoundsUpdateProb);

        for iBeat = 1:oReconParas.nBeats
            for iSubBeatM = 1:oReconParas.nSubBeatsM
                % CBED data
                cbeds1SubBeatMG = oCBEDReconG.getSubBeatG(iBeat, iSubBeatM, oReconParas.nSubSpotsParallelM);

                [cys1SubBeatM, cxs1SubBeatM, ~, ~] = oSpot.getSubBeatScanCoors(iBeat, iSubBeatM, oReconParas.nSubSpotsParallelM); % [m] (nSubSpotsParallelM, 1)     
                oReconMG.multisliceForwardingGPU(cys1SubBeatM, cxs1SubBeatM, oObjG, oProbG);
                oReconMG.calGradFGPU(cbeds1SubBeatMG);
                oReconMG.calGradRGPU();
                oReconMG.mergeGradsGPU(cys1SubBeatM, cxs1SubBeatM);
            end % end for iSubBeatM
            oReconMG.updateGPU(oObjG, oProbG, updateObj, updateProb);
            oReconMG.zerolizeGradAll();
        end % end for iBeat
        
        % III.3 postprocessing
        if oReconParas.isShrinkageObjOn
            oReconMG.shrinkageObjGPU(oObjG, object_shrinkage_maskG);
        end
        if oReconParas.isCenterizeProb
            oProbG.CoM_centerize();
        end
    
        % III.4 save outputs
        isSaveOutputs = mod(iRound, oReconParas.everyNRoundsToSaveOutputs) == 0;
        if isSaveOutputs
            if updateProb
                path_probAbs = fullfile(oReconParas.path_saveOutputs, ['probAbs', num2str(iRound, '%03d'), '.mrcs']);
                path_probAngle = fullfile(oReconParas.path_saveOutputs, ['probAngle', num2str(iRound, '%03d'), '.mrcs']);
                oProbG.save(path_probAbs, path_probAngle);
            end
            switch oReconParas.objModel
                case 'potential'
                    path_objPot = fullfile(oReconParas.path_saveOutputs, ['objAngle', num2str(iRound, '%03d'), '.mrcs']);
                    oObjG.savePot(path_objPot);
                case 'complex'
                    path_objAbs = fullfile(oReconParas.path_saveOutputs, ['objAbs', num2str(iRound, '%03d'), '.mrcs']);
                    path_objAngle = fullfile(oReconParas.path_saveOutputs, ['objAngle', num2str(iRound, '%03d'), '.mrcs']);
                    oObjG.save(path_objAbs, path_objAngle);
                otherwise
                    error('Wrong object model!');
            end
        end
        
        % III.5 analysis the likelihoods
        likelihoodLine = 'o';
        if oReconParas.isMonitorLikelihoods && isSaveOutputs
            negativeLoglikelihoods = zeros(oReconParas.nSpotsParallel, oReconParas.nBeats, 'single');
            for iBeat = 1:oReconParas.nBeats
                for iSubBeatL = 1:oReconParas.nSubBeatsL
                    [lind, uind] = oSpot.subBeatInds(iSubBeatL, oReconParas.nSubSpotsParallelL);
                    cbeds1SubBeatLG = oCBEDReconG.getSubBeatG(iBeat, iSubBeatL, oReconParas.nSubSpotsParallelL); % (nYX, nYX, 1, nSpotsParallel) gpuArray
                    [cys1SubBeatL, cxs1SubBeatL, ~, ~] = oSpot.getSubBeatScanCoors(iBeat, iSubBeatL, oReconParas.nSubSpotsParallelL); % (nSpotsParallel, 1)
                    negativeLoglikelihoods(lind:uind, iBeat) = oReconLG.calNegativeLogLikelihoodGPU(cys1SubBeatL, cxs1SubBeatL, ...
                        oObjG, oProbG, ...
                        cbeds1SubBeatLG, ...
                        maskFMonitorLikelihoodsG);
                end
            end
            validNegativeLoglikehoods = negativeLoglikelihoods;
            meanLikelihood = mean(validNegativeLoglikehoods, 'all');
            maxLikelihood = max(validNegativeLoglikehoods, [], 'all');
            minLikelihood = min(validNegativeLoglikehoods, [], 'all');
            stdLikelihood = std(validNegativeLoglikehoods, 0, 'all');
            likelihoodLine = ['Likelihood: ', num2str(meanLikelihood, '%.6f'), ...
                ' (min: ', num2str(minLikelihood, '%.6f'), ...
                ', max: ', num2str(maxLikelihood, '%.6f'), ...
                ', std: ', num2str(stdLikelihood, '%.6f'), ')'];
        end

        elapsedTimeLine = 'o'; 
        if oReconParas.isTicToc
            % timer 1 end
            wait(gpuDev);
            elapsedTimeLine = ['Elapsed Time: ', num2str(toc(timer1), '%.2f'), 's'];
        end

        % III.6 output logs
        io.DispAndLog(logFileID, [num2str(iRound), '/', num2str(oReconParas.nRounds), ...
            ' | ', elapsedTimeLine, ' | ', likelihoodLine]);
        
    end % end for iRound
    disp('Bravo! Cool!');
end