classdef cReconProSimplifiedG < handle
    % reconstruction algorithm of GPU version
    properties
        objModel % 'potential' or 'complex'
        % FFT
        % ifftshiftIdx % cell, (1, 2)
        % fftshiftIdx % cell, (1, 2)
        % faster version of matlab fftshift/ifftshift to work for stack of 2D images (inspired by ChenZ's Code) data(fftshiftIdx{:}, :, :, ...)
        phaseModG % fft2 + (.* phaseModG) == ifftshift + fft2
        phaseInvModG % (.* phaseInvModG) + ifft2 == ifft2 + fftshift

        % --- data blocks for forwarding ---
        % reconstruction or sampling
        midObjWavesG % intermediate object functions (nYX, nYX, nSpotsParallel), complex, GPU
        cbedsCalG % calculated cbeds (nYX, nYX, nSpotsParallel), real, GPU
        midIcdWavesG % intermediate incident wave functions (nYX, nYX, nSpotsParallel), complex, GPU
        midDiffsG % diffraction wave functions (nYX, nYX, nSpotsParallel), complex, GPU

        % --- data blocks for backwarding & SGD ---
        midChiWavesG % the gradient of the exit wave functions (nYX, nYX, nSpotsParallel), complex, GPU
        % masks
        maskGradFG % (nYX, nYX), real, GPU, mask fourier space in calGradF
        % local grads
        objGradsPotG % (nYX, nYX, nSpotsParallel), ** single **, GPU
        objGradsCG % (nYX, nYX, nSpotsParallel), ** complex **, GPU
        probGradsG % (nYX, nYX, nSpotsParallel), complex, GPU
        % step size factor, from 0 to 1, alpha for object, beta for probe
        stepSizeFactor
        % global grads
        oObjGradAllPotG % a cObjG object representing objGradAll, ** single **, GPU
        oObjGradAllCG % a cObjG object representing objGradAll, ** complex **, GPU
        object_normalizationG % single, GPU, (nYX, nYX, nModes)
        oProbGradAllG % a cProbG object representing probGradAll, complex, GPU
        oObj_probe_normalizationG % single, GPU

        % size features related on data blocks
        nYX % (probN)
        nSpotsParallel

        status % -1: uninitialized, 0: forwarding only, 1: forwarding & backwarding for sampling, 
        % 2: forwarding & backwarding for sampling & backwarding for reconstruction,

    end
    methods
        % initialization
        function obj = cReconProSimplifiedG()
            obj.status = -1; % uninitialized
        end
        function initForwarding(obj, objModel, probN, nSpotsParallel)
            % set the size features
            obj.objModel = objModel;
            obj.nYX = probN;
            obj.nSpotsParallel = nSpotsParallel;
            
            % fft-related
            [U, V] = meshgrid(single(0:(obj.nYX-1)));
            if mod(obj.nYX, 2) == 0 % even
                obj.phaseModG = gpuArray(exp(1i * pi * (U + V)));
            else % odd
                obj.phaseModG = gpuArray(exp(1i * pi * (obj.nYX - 1) / obj.nYX * (U + V)));
            end
            obj.phaseInvModG = conj(obj.phaseModG);

            % --- initialize data blocks for forwarding ---
            obj.midObjWavesG = complex(zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'), ...
                zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'));
            obj.cbedsCalG = zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray');
            obj.midIcdWavesG = complex(zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'), ...
                zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'));
            obj.midDiffsG = complex(zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'), ...
                zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'));

            obj.status = 0;
        end
        function initBackwardingPartA(obj, gradFMask)
            % forwarding & backwarding for sampling
            if obj.status ~= 0 && obj.status ~= 1
                error('Please call initForwarding() first!');
            end

            % --- initialize data blocks for backwarding & SGD ---
            obj.midChiWavesG = complex(zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'), ...
                zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'));
            % masks
            obj.maskGradFG = gpuArray(ifftshift(ifftshift(gradFMask, 1), 2));
            % local grads
            obj.objGradsCG = complex(zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'), ...
                zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'));
            if isequal(obj.objModel, 'potential')
                obj.objGradsPotG = zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray');
            end
            obj.probGradsG = complex(zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'), ...
                zeros(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray'));

            obj.status = 1;
        end
        function initBackwardingPartB(obj, objNY, objNX, pSize, oriIndY, oriIndX, stepSizeFactor)
            % forwarding & backwarding for sampling & backwarding for reconstruction
            if obj.status ~= 1 && obj.status ~= 2
                error('Please call initBackwardingPartA() first!');
            end
            
            % step size factor, alpha for object, beta for probe
            obj.stepSizeFactor = stepSizeFactor; % (alpha, beta)

            % global grads
            switch obj.objModel
                case 'potential'
                    obj.oObjGradAllPotG = objects.cObjG;
                    obj.oObjGradAllPotG.initZeros(objNY, objNX, 1, ...
                        pSize, 0.0, oriIndY, oriIndX, ...
                        'single'); % set the accumulated variable to 0.0
                case 'complex'
                    obj.oObjGradAllCG = objects.cObjG;
                    obj.oObjGradAllCG.initZeros(objNY, objNX, 1, ...
                        pSize, 0.0, oriIndY, oriIndX, ...
                        'complex'); % set the accumulated variable to 0.0
                otherwise
                    error('Wrong object model!');
            end
            obj.oProbGradAllG = probes.cProbeG;
            obj.oProbGradAllG.probG = complex(zeros(obj.nYX, obj.nYX, 1, 'single', 'gpuArray'), ...
                zeros(obj.nYX, obj.nYX, 1, 'single', 'gpuArray')); % set the accumulated variable to 0.0
            obj.oProbGradAllG.initUVG();
            % other parameters like wavelength in oProbGradAllG are irrelevant
            
            % normalization
            obj.object_normalizationG = zeros(obj.nYX, obj.nYX, 1, 'single', 'gpuArray');
            obj.oObj_probe_normalizationG = objects.cObjG;
            obj.oObj_probe_normalizationG.initZeros(objNY, objNX, 1, ...
                pSize, 0.0, oriIndY, oriIndX, ...
                'single'); % set the accumulated variable to 0.0

            obj.status = 2;
        end
        
        % --- 7 steps in reconstruction ---
        % 1. multislice forward
        multisliceForwardingGPU(obj, cys, cxs, oObjG, oProbG);

        % 2. calculate gradient in fourier space
        calGradFGPU(obj, cbeds1BeatG);

        % 3. no multislice backward (calculate gradients of the exit wave on each slice)
        
        % 4. calculate gradient in real space (-grad)
        calGradRGPU(obj);
        
        % 5. no calculate the step size

        % 6. merge the grads of different spots
        function  zerolizeGradAll(obj)
            % set the accumulated variable to 0.0
            switch obj.objModel
                case 'potential'
                    obj.oObjGradAllPotG.zerolize();
                case 'complex'
                    obj.oObjGradAllCG.zerolize();
                otherwise
                    error('Wrong object model!');
            end
            obj.oProbGradAllG.zerolize();
            obj.object_normalizationG = obj.object_normalizationG * zeros('like', obj.object_normalizationG);
            obj.oObj_probe_normalizationG.zerolize();
        end

        mergeGradsGPU(obj, cys, cxs);

        % 7. update oObj and oProb
        updateGPU(obj, oObjG, oProbG, updateObj, updateProb);
        
        % post-precess: 
        % 2. shrinkage low-illumination area
        shrinkageObjGPU(obj, oObjG, object_shrinkage_maskG);

        % calculate negative log likelihoods
        negativeLoglikelihoods = calNegativeLogLikelihoodGPU(obj, cys, cxs, oObj, oProb, oCBEDReconG, iBeat, maskFLikelihoodG);
        % cys/cxs: [m] (nSpotsParallel, 1), negativeLoglikelihoods (nSpotsParallel, 1), cpu

    end
end