classdef cPtychoReconParas < handle
    properties
        % outputs
        path_saveOutputs % path to save the outputs(objAbsX.mrcs, objAngleX.mrcs, probAbsX.mrcs, probAngleX.mrcs)

        % images
        updownsamplingF % up(>1)/down-sampling in fourier space, reduce/enlarge pSizeF
        updownsamplingR % up(>1)/down-sampling in real space, enlarge/reduce nyquist frequency
        probN % number of pixels in each image, probN can be larger than diffN using super-resolution technique
        pSizeF % [1/m] the fourier pixel size used in the reconstruction
        pSize % [m] determined by probN and pSizeF

        % scan positions
        % - crop a subArea to reconstruct -
        subAreaScanIDs % the list of the indice of the scan to be reconstructed
        subAreaScanN % the total number of scans in subArea
        subArea % the side length of the subArea [m], in which all the scans are covered
        subAreaScanCoors % coordinates of all the scan spots
        % (subAreaScanN, 2) 2:[m] (y, x)
        % (y, x) is the scan spot's coordinate,
        % with the origin at the center of the subArea
        isRecenterAfterCrop % when calculate FRC, please set it to false
        % - add random offset (gaussian noise) to the coordinates of the scans -
        scanPerturbSigma % [m]

        % cbed
        path_cbeds % path to the mrc file of cbeds, the input data of the reconstruction process
        flipMode
        % flip: 0-no flip, 1-flip vertically, 2-flip horizontally, 3-flip vertically + flip horizontally 
        % 4-transpose, 5-transpose+flip vertically, 6-transpose+flip horizontally, 7-transpose+flip vertically+flip horizontally
        nZeroEdges % set nZeroEdges rows/cols at the edge of CBED to 0.0
        cbedTrans % pix [tx, ty], moving torwards right/down is positive, size: (1, 2) or (subAreaScanN, 2)
        cbedMask
        isLoadingCBED
        savedCBEDFile
        isSavingCBED
        isCBEDOnGPU

        % obj
        objModel % 'complex' or 'potential'
        objNY
        objNX
        oriIndY % the index of the origin of the obj along Y axis
        oriIndX % the index of the origin of the obj along X axis
        sliThick % [m] thickness of each slice
        nSlices
        path_objToLoad % {'path to abs', 'path to angle'}
        isLoadObj
        nPadForLoad

        % multi-modes of prob
        upsamplingProbFFactor % >= 1, upsampling in fourier space, enlarge the imgSizeR, then cropping in real space
        nModes % number of modes of the probe
        slaveModeWeight % slaveMode's intensity: totalIntensity * slaveModeWeight, 
            % slaveModeWeight: [0, 1）
            % if there is only base probe, set slaveModeWeight to 0.0
        path_vaccumProbe
        isInitUsingVacuumProbe
        path_probToLoad % {'path to abs', 'path to angle'}
        isLoadProb
        isAdjustProbIntensity
        
        % iteration
        nRounds % the number of rounds in reconstruction
        iRoundsUpdateObj % in which round to update obj
        iRoundsUpdateProb % in which round to update prob
        everyNRoundsToSaveOutputs
        
        % beats
        % 1 beat involves nSpotsParallel spots, 1 round involves nBeats beats
        nSpotsParallel % these spots are calculated simultaneously
        nBeats % these beats are calculated sequentially
        beats % the spot IDs, (nSpotsParallel, nBeats)
        path_beatToLoad
        
        % recon algorithm setting
        nSubSpotsParallelM % in case of cuda memory is not sufficient, each beat may be split into several subbeats
        nSubBeatsM
        stepSizeFactor % a factor plus stepsize, [alpha, beta], alpha -> object, beta -> probe
        gradFMask % for masking fourier space in calGradF
        
        % postprocessing
        isShrinkageObjOn % whether to turn on the shrinkageObj
        isCenterizeProb % whether to centerize the probe function (CoM) at each iteration

        % monitoring log-likelihood
        isMonitorLikelihoods
        nSubSpotsParallelL
        nSubBeatsL
        monitorLikelihoodsMask

        % control
        isTicToc
        saveInit
        
    end
    methods
        % initialization
        function init(obj, ...
                oPtychoExpParas, ... % the object of cPtychoExpParas
                cfg_in)
            % cfg_in: a struct to hold all the input parameters

            % outputs
            obj.path_saveOutputs = cfg_in.path_saveOutputs;

            % images
            obj.updownsamplingF = cfg_in.updownsamplingF;
            obj.updownsamplingR = cfg_in.updownsamplingR;
            obj.probN = ceil(ceil(oPtychoExpParas.diffN * obj.updownsamplingF) * obj.updownsamplingR);
            obj.pSizeF = (oPtychoExpParas.pSizeFOrigin * oPtychoExpParas.diffN) / ceil(oPtychoExpParas.diffN * obj.updownsamplingF);
            obj.pSize = 1 / obj.probN / obj.pSizeF;
            
            % scan positions
            % - crop a subArea -
            obj.subAreaScanIDs = cfg_in.subAreaScanIDs;
            obj.subAreaScanN = length(obj.subAreaScanIDs);
            subAreaScanCoorsOld = oPtychoExpParas.scanCoors(obj.subAreaScanIDs, :); % (subAreaScanN, 2)
            % - add random offset (gaussian noise) to the coordinates of the scans -
            obj.scanPerturbSigma = cfg_in.scanPerturbSigma;
            subAreaScanCoorsOld = subAreaScanCoorsOld + obj.scanPerturbSigma * randn(size(subAreaScanCoorsOld));
            minY = min(subAreaScanCoorsOld(:, 1));
            maxY = max(subAreaScanCoorsOld(:, 1));
            minX = min(subAreaScanCoorsOld(:, 2));
            maxX = max(subAreaScanCoorsOld(:, 2));
            centerY = (maxY + minY) / 2;
            centerX = (maxX + minX) / 2;
            obj.isRecenterAfterCrop = cfg_in.isRecenterAfterCrop;
            if obj.isRecenterAfterCrop
                obj.subArea = max(maxY - minY, maxX - minX); % [m], a square object function
                obj.subAreaScanCoors = subAreaScanCoorsOld - [centerY, centerX]; % (subAreaScanN, 2)
            else
                obj.subArea = 2 * (max(maxY - minY, maxX - minX) / 2 + max(abs(centerY), abs(centerX)));
                obj.subAreaScanCoors = subAreaScanCoorsOld;
            end
            
            % cbed
            obj.path_cbeds = cfg_in.path_cbeds;
            obj.flipMode = cfg_in.flipMode;
            obj.nZeroEdges = cfg_in.nZeroEdges;
            obj.cbedTrans = cfg_in.cbedTrans;
            obj.cbedMask = cfg_in.cbedMask;
            obj.isLoadingCBED = cfg_in.isLoadingCBED;
            obj.isSavingCBED = cfg_in.isSavingCBED;
            obj.savedCBEDFile = cfg_in.savedCBEDFile;
            obj.isCBEDOnGPU = cfg_in.isCBEDOnGPU;

            % obj
            obj.objModel = cfg_in.objModel;
            isSetObjN = cfg_in.objNYXSet > 0;
            if isSetObjN
                obj.objNY = cfg_in.objNYXSet;
                obj.objNX = cfg_in.objNYXSet;
                objN = 1 + ceil(obj.probN + obj.subArea / obj.pSize) + 1;
                if cfg_in.objNYXSet < objN
                    error(['objNYXSet is too small!!! At least ', num2str(objN), ' pix']);
                end
            else
                objN = cfg_in.nPad + ceil(obj.probN + obj.subArea / obj.pSize) + cfg_in.nPad;
                obj.objNY = objN;
                obj.objNX = objN;
            end
            obj.oriIndY = floor(obj.objNY / 2) + 1;
            obj.oriIndX = floor(obj.objNX / 2) + 1;
            obj.sliThick = cfg_in.sliThick;
            obj.nSlices = cfg_in.nSlices;
            obj.path_objToLoad = cfg_in.path_objToLoad;
            if isequal(obj.objModel, 'potential')
                obj.isLoadObj = ~isempty(obj.path_objToLoad{2});
            elseif isequal(obj.objModel, 'complex')
                obj.isLoadObj = ~isempty(obj.path_objToLoad{1}) & ~isempty(obj.path_objToLoad{2});
            else
                error("objModel is wrong!");
            end
            obj.nPadForLoad = cfg_in.nPadForLoad;

            % multi-modes of prob
            % - initialize from scratch -
            obj.upsamplingProbFFactor = cfg_in.upsamplingProbFFactor;
            obj.nModes = cfg_in.nModes;
            obj.slaveModeWeight = cfg_in.slaveModeWeight;
            if obj.nModes == 1
                obj.slaveModeWeight = 0.0;
            end
            obj.path_vaccumProbe = cfg_in.path_vaccumProbe;
            obj.isInitUsingVacuumProbe = ~isempty(obj.path_vaccumProbe);
            % - load from file -
            obj.path_probToLoad = cfg_in.path_probToLoad;
            obj.isLoadProb = (~isempty(obj.path_probToLoad{1})) & (~isempty(obj.path_probToLoad{2}));
            obj.isAdjustProbIntensity = cfg_in.isAdjustProbIntensity;

            % iteration
            obj.nRounds = cfg_in.nRounds;
            obj.iRoundsUpdateObj = cfg_in.iRoundsUpdateObj;
            obj.iRoundsUpdateProb = cfg_in.iRoundsUpdateProb;
            obj.everyNRoundsToSaveOutputs = cfg_in.everyNRoundsToSaveOutputs;

            % beats
            obj.path_beatToLoad = cfg_in.path_beatToLoad;
            isLoadBeat = ~isempty(obj.path_beatToLoad);
            if isLoadBeat
                tmp = load(obj.path_beatToLoad); % (.mat)
                obj.beats = tmp.beats; % (nSpotsParallel, nBeats)
                obj.nSpotsParallel = size(obj.beats, 1);
                obj.nBeats = size(obj.beats, 2);
                clear tmp;
            else
                obj.nSpotsParallel = cfg_in.nSpotsParallel;
                obj.nBeats = ceil(obj.subAreaScanN / obj.nSpotsParallel);
                spotIDs = [1:obj.subAreaScanN, ...
                    randi(obj.subAreaScanN, ...
                    1, obj.nSpotsParallel * obj.nBeats - obj.subAreaScanN) ...
                    ];
                spotIDs = spotIDs(randperm(obj.nSpotsParallel * obj.nBeats));
                obj.beats = reshape(spotIDs, obj.nSpotsParallel, obj.nBeats);
            end

            % recon algorithm setting
            obj.nSubSpotsParallelM = cfg_in.nSubSpotsParallelM;
            if mod(obj.nSpotsParallel, obj.nSubSpotsParallelM) ~= 0
                error('Do not support undividable nSubSpotsParallelM');
            end
            obj.nSubBeatsM = obj.nSpotsParallel / obj.nSubSpotsParallelM;
            obj.stepSizeFactor = cfg_in.stepSizeFactor;
            obj.gradFMask = cfg_in.gradFMask;

            % postprocessing
            obj.isShrinkageObjOn = cfg_in.isShrinkageObjOn;
            obj.isCenterizeProb = cfg_in.isCenterizeProb;

            % monitoring log-likelihood
            obj.isMonitorLikelihoods = cfg_in.isMonitorLikelihoods;
            obj.nSubSpotsParallelL = cfg_in.nSubSpotsParallelL;
            if mod(obj.nSpotsParallel, obj.nSubSpotsParallelL) ~= 0
                error('Do not support undividable nSubSpotsParallelL');
            end
            obj.nSubBeatsL = obj.nSpotsParallel / obj.nSubSpotsParallelL;
            obj.monitorLikelihoodsMask = cfg_in.monitorLikelihoodsMask;

            % control
            obj.isTicToc = cfg_in.isTicToc;
            obj.saveInit = cfg_in.saveInit;
        end

    end
end