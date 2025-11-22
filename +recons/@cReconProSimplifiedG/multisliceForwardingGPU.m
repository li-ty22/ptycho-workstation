function multisliceForwardingGPU(obj, cys, cxs, oObjG, oProbG)
% cys/cxs: [m] (nSpotsParallel, 1) coordinates of the spots

    % perpare data for multislice forwarding
    % objWaves of different spots, (nYX, nYX, 1, nSpotsParallel, nSlices)
    switch obj.objModel
        case 'potential'
            obj.midObjWavesG = ones(obj.nYX, obj.nYX, obj.nSpotsParallel, 'single', 'gpuArray') .* ...
                exp(1i * oObjG.cropVolumesGPU(cys, cxs, obj.nYX, 'isInterpolation', false)); % (nYX, nYX, nSpotsParallel)
        case 'complex'
            obj.midObjWavesG = oObjG.cropVolumesGPU(cys, cxs, obj.nYX, 'isInterpolation', false); % (nYX, nYX, nSpotsParallel)
        otherwise
            error('Wrong object model!');
    end
    [~, ~, disYs, disXs] = oObjG.coordConversion(cys, cxs, obj.nYX);
    % probWaves of different spots, (nYX, nYX, nSpotsParallel)
    obj.midIcdWavesG = oProbG.getTranProbsGPUSimplified(disYs, disXs);

    obj.midDiffsG = fft2(obj.midIcdWavesG .* obj.midObjWavesG) .* obj.phaseModG; % (nYX, nYX, nSpotsParallel)
    % with the zero-frequency component at the upper-left corner

    obj.cbedsCalG = abs(obj.midDiffsG).^2; % (nYX, nYX, nSpotsParallel)
end