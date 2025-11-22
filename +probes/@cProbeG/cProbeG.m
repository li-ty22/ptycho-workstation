% represent the probe function on GPU
classdef cProbeG < handle
    properties
        probG % (probN, probN, nModes) images of probes in different modes, GPU
        pSize % [m]
        % parameters used in initialization
        waveLength % [m]
        defocus % [m]
        Cs % [m]
        semiAngle % [rad]

        % for getTranProbsGPU
        UG
        VG
    end
    methods
        % initialization
        % 1-1. init the base probe
        initBase(obj, probN, pSizeF, waveLength, defocus, Cs, semiAngle, vacuumProbe, varargin);
        % 1-2. init multiple modes via Laguerre-Gaussian, call initBase() first
        initMultiModes(obj, nModes);
        % 2. init from data
        function initFromData(obj, probIn, pSize, waveLength, defocus, Cs, semiAngle)
            if isgpuarray(probIn)
                obj.probG = single(probIn);
            else
                obj.probG = gpuArray(single(probIn));
            end
            obj.pSize = pSize;
            obj.waveLength = waveLength;
            obj.defocus = defocus;
            obj.Cs = Cs;
            obj.semiAngle = semiAngle;
            obj.initUVG();
        end

        % utils
        function initUVG(obj)
            center = floor(obj.nYX() / 2 + 1);
            [U, V] = meshgrid(1:obj.nYX());
            obj.UG = gpuArray(ifftshift(U - center)) / obj.nYX(); % (nYX, nYX)
            obj.VG = gpuArray(ifftshift(V - center)) / obj.nYX(); % (nYX, nYX)
        end

        % set probG to 0.0 (set the accumulated variable to 0.0)
        function zerolize(obj)
            obj.probG = obj.probG * zeros('like', obj.probG);
        end

        % orthogonalize the different modes
        function orthogonalizeGPU(obj)
            % each mode's intensity turns to 1.0, need to normalize manually
            vecProbG = reshape(obj.probG, [obj.nYX()^2, obj.nModes()]);
            [orthogonalVecProbG, ~] = utils.mgsGPU(vecProbG); % modified Gram-Schmidt method
            obj.probG = reshape(orthogonalVecProbG, [obj.nYX(), obj.nYX(), obj.nModes()]);
        end

        function orthogonalizeAfterUpdateGPU(obj)
            inten = sqrt(sum(abs(obj.probG).^2, [1 2]));
            obj.probG = obj.probG ./ inten;
            obj.orthogonalizeGPU();
            for iMode = 1:obj.nModes()
                aProbG = obj.probG(:, :, iMode);
                obj.probG(:, :, iMode) = aProbG / sqrt(sum(abs(aProbG).^2, [1 2])) * inten(iMode);
            end
        end

        % adjust the intensity
        function normalizeGPU(obj, totalIntensity, slaveModeWeight)
            % slaveMode's intensity: totalIntensity * slaveModeWeight
            % slaveModeWeight: [0, 1）
            % if there is only base probe, set slaveModeWeight to 0.0
            baseProbTotalIntensity = totalIntensity * (1.0 - (obj.nModes()-1) * slaveModeWeight);
            slaveModeTotalIntensity = totalIntensity * slaveModeWeight;
            % base
            aProbG = obj.getProbImGPU(1);
            obj.probG(:, :, 1) = obj.probG(:, :, 1) / sqrt(sum(abs(aProbG).^2, 'all')) * sqrt(baseProbTotalIntensity);
            % slave
            for iMode = 2:obj.nModes()
                aProbG = obj.getProbImGPU(iMode);
                obj.probG(:, :, iMode) = obj.probG(:, :, iMode) / sqrt(sum(abs(aProbG).^2, 'all')) * sqrt(slaveModeTotalIntensity);
            end
        end

        % get the probe of mode iMode, return a GPU array
        function probImgG = getProbImGPU(obj, iMode)
            probImgG = obj.probG(:, :, iMode);
        end
        % get the probe of mode iMode, return a Matlab array
        function probImg = getProbIm(obj, iMode)
            probImg = gather(obj.probG(:, :, iMode));
        end
        % get the sum of probe intensity of all the modes
        function probIntensityG = getProbIntensityGPU(obj)
            probIntensityG = sum(abs(obj.probG).^2, 3);
        end

        % fourier interpolation
        function tranProbsG = getTranProbsGPU(obj, tranys_pix, tranxs_pix)
            % tranys_pix/tranxs_pix: [pix] (nSpotsParallel, 1)
            % tranProbsG: (nYX, nYX, nModes, nSpotsParallel), gpu
            
            % % no interpolation
            % tranProbsG = repmat(obj.probG, [1 1 1 numel(tranys_pix)]);

            % fourier interpolation
            phaseRampG = exp(-1i * 2 * pi * ...
                ( ...
                obj.UG .* reshape(tranxs_pix, 1, 1, 1, numel(tranxs_pix)) + ...
                obj.VG .* reshape(tranys_pix, 1, 1, 1, numel(tranys_pix)) ...
                ) ...
                ); % (nYX, nYX, 1, nSpotsParallel) complex
            % down/right: +
            tranProbsG = ifft2(fft2(obj.probG) .* phaseRampG);
        end
        function tranProbsG = getTranProbsGPUSimplified(obj, tranys_pix, tranxs_pix)
            % tranys_pix/tranxs_pix: [pix] (nSpotsParallel, 1)
            % tranProbsG: (nYX, nYX, nSpotsParallel), gpu
            
            % % no interpolation
            % tranProbsG = repmat(obj.probG(:, :, 1), [1 1 numel(tranys_pix)]);

            % fourier interpolation
            phaseRampG = exp(-1i * 2 * pi * ...
                ( ...
                obj.UG .* reshape(tranxs_pix, 1, 1, numel(tranxs_pix)) + ...
                obj.VG .* reshape(tranys_pix, 1, 1, numel(tranys_pix)) ...
                ) ...
                ); % (nYX, nYX, nSpotsParallel) complex
            % down/right: +
            tranProbsG = ifft2(fft2(obj.probG(:, :, 1)) .* phaseRampG);
        end
        function [gradTranys_pix, gradTranxs_pix] = getGradTranYXsPixGPU(obj, gradProbsG, tranys_pix, tranxs_pix)
            % tranys_pix / tranxs_pix: [pix] (nSpotsParallel, 1)
            % gradProbsG: (nYX, nYX, nModes, nSpotsParallel), gpu
            % gradTranys_pix, gradTranxs_pix: (nSpotsParallel, 1)
            nSpotsParallel = numel(tranys_pix);
            phaseRampG = exp(-1i * 2 * pi * ...
                ( ...
                obj.UG .* reshape(tranxs_pix, 1, 1, 1, numel(tranxs_pix)) + ...
                obj.VG .* reshape(tranys_pix, 1, 1, 1, numel(tranys_pix)) ...
                ) ...
                ); % (nYX, nYX, 1, nSpotsParallel) complex
            % down/right: +
            grad1G = imag(fft2(gradProbsG) .* conj(fft2(obj.probG)) .* conj(phaseRampG)); % (nYX, nYX, nModes, nSpotsParallel)
            gradTranys_pix = gather(reshape(sum(grad1G .* (-2 * pi * obj.VG), [1 2 3]), nSpotsParallel, 1));
            gradTranxs_pix = gather(reshape(sum(grad1G .* (-2 * pi * obj.UG), [1 2 3]), nSpotsParallel, 1));
        end
        function fuseAddTranProbsGPU(obj, tranProbsG, tranys_pix, tranxs_pix, weightsG)
            % tranys_pix/tranxs_pix: [pix] (nSpotsParallel, 1)
            % tranProbsG: (nYX, nYX, nModes, nSpotsParallel), gpu
            % weightsG: (nSpotsParallel, 1), gpu
            
            % % no interpolation
            % obj.probG = obj.probG + sum(tranProbsG .* reshape(weightsG, 1, 1, 1, numel(weightsG)), 4);

            % fourier interpolation
            invPhaseRampG = exp(-1i * 2 * pi * ...
                ( ...
                obj.UG .* reshape(- tranxs_pix, 1, 1, 1, numel(tranxs_pix)) + ...
                obj.VG .* reshape(- tranys_pix, 1, 1, 1, numel(tranys_pix)) ...
                ) ...
                ); % (nYX, nYX, 1, nSpotsParallel) complex
            % down/right: +
            addProbsG = ifft2(fft2(tranProbsG) .* invPhaseRampG); % (nYX, nYX, nModes, nSpotsParallel)
            obj.probG = obj.probG + sum(addProbsG .* reshape(weightsG, 1, 1, 1, numel(weightsG)), 4);
        end
        function fuseAddTranProbsGPUSimplified(obj, tranProbsG, tranys_pix, tranxs_pix, weightsG)
            % tranys_pix/tranxs_pix: [pix] (nSpotsParallel, 1)
            % tranProbsG: (nYX, nYX, nSpotsParallel), gpu
            % weightsG: (nSpotsParallel, 1), gpu
            
            % % no interpolation
            % obj.probG = obj.probG + sum(tranProbsG .* reshape(weightsG, 1, 1, numel(weightsG)), 3);

            % fourier interpolation
            invPhaseRampG = exp(-1i * 2 * pi * ...
                ( ...
                obj.UG .* reshape(- tranxs_pix, 1, 1, numel(tranxs_pix)) + ...
                obj.VG .* reshape(- tranys_pix, 1, 1, numel(tranys_pix)) ...
                ) ...
                ); % (nYX, nYX, nSpotsParallel) complex
            % down/right: +
            addProbsG = ifft2(fft2(tranProbsG) .* invPhaseRampG); % (nYX, nYX, nSpotsParallel)
            obj.probG(:, :, 1) = obj.probG(:, :, 1) + sum(addProbsG .* reshape(weightsG, 1, 1, numel(weightsG)), 3);
        end

        % obtain the probe's radius[pix] (rms)
        function radius = calRMSRadiusGPU(obj)
            sqrtProbAllModeIntensityG = sqrt(obj.getProbIntensityGPU());
            th = rms(sqrtProbAllModeIntensityG, 'all'); % root mean square
            maskG = single(sqrtProbAllModeIntensityG > th);
            radius = gather(sqrt(sum(maskG, 'all') / pi));
        end
        % obtain the probe's radius[pix] (1 percent intensity)
        function radius = cal1PercentIntensityRadiusGPU(obj)
            th = 0.01 * max(obj.getProbIntensityGPU(), [], 'all');
            maskG = single(obj.getProbIntensityGPU() > th);
            radius = gather(sqrt(sum(maskG, 'all') / pi));
        end

        % calculate the radial CDF of the intensity
        function radialCDF = calIntensityRadialCDFGPU(obj)
            rA = utils.radialSum(obj.getProbIntensityGPU(), 40);
            radialCDF = cumsum(rA);
        end

        % nYX
        function n = nYX(obj)
            n = size(obj.probG, 1);
        end
        % nModes
        function n = nModes(obj)
            n = size(obj.probG, 3);
        end

        function CoM_centerize(obj)
            probIntensityG = abs(obj.probG).^2; % (nYX, nYX, nModes)
            M = sum(probIntensityG, [1 2]); % (1, 1, nModes)
            % calculate the mass center
            center = floor(obj.nYX() / 2) + 1;
            [X, Y] = meshgrid(1:obj.nYX());
            m_x = sum((X - center) .* probIntensityG, [1 2]) ./ M; % (1, 1, nModes)
            m_y = sum((Y - center) .* probIntensityG, [1 2]) ./ M; % (1, 1, nModes)
            % tran
            [U, V] = meshgrid(1:obj.nYX());
            UG = gpuArray(ifftshift(U - center)) / obj.nYX(); % (nYX, nYX)
            VG = gpuArray(ifftshift(V - center)) / obj.nYX(); % (nYX, nYX)
            phaseRampG = exp(-1i * 2 * pi * ...
                ( ...
                UG .* (-m_x) + ...
                VG .* (-m_y) ...
                ) ...
                ); % (nYX, nYX, nModes) complex
            % down/right: +
            obj.probG = ifft2(fft2(obj.probG) .* phaseRampG);
        end

        % output
        function save(obj, absMRCFile, angleMRCFile)
            pSize_A = obj.pSize * 1e10; % [A]
            prob = gather(obj.probG);
            isSaveAbs = ~isempty(absMRCFile);
            isSaveAngle = ~isempty(angleMRCFile);
            if isSaveAbs
                io.WriteMRC(abs(prob), pSize_A, absMRCFile, 2, obj.nModes());
            end
            if isSaveAngle
                io.WriteMRC(angle(prob), pSize_A, angleMRCFile, 2, obj.nModes());
            end
            if ~isSaveAbs && ~isSaveAngle
                warning('No file was saved!');
            end
        end

    end
end