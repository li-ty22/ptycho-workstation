function mergeGradsGPU(obj, cys, cxs)
    % cys/cxs: (nSpotsParallel, 1) coordinates of the spots
    
    % spotsWeightObj/ProbG are abandoned
    weights = ones(numel(cys), 1, 'single', 'gpuArray');

    % object
    switch obj.objModel
        case 'potential'
            obj.oObjGradAllPotG.fuseAddVolumesGPU(obj.objGradsPotG .* obj.stepSizeFactor(1), cys, cxs, weights, 'isInterpolation', false);
        case 'complex'
            obj.oObjGradAllCG.fuseAddVolumesGPU(obj.objGradsCG .* obj.stepSizeFactor(1), cys, cxs, weights, 'isInterpolation', false);
        otherwise
            error('Wrong object model!');
    end
    obj.oObj_probe_normalizationG.fuseAddVolumesGPU(abs(obj.midIcdWavesG).^2, cys, cxs, weights, 'isInterpolation', false);

    % probe
    switch obj.objModel
        case 'potential'
            [~, ~, disYs, disXs] = obj.oObjGradAllPotG.coordConversion(cys, cxs, obj.nYX);
        case 'complex'
            [~, ~, disYs, disXs] = obj.oObjGradAllCG.coordConversion(cys, cxs, obj.nYX);
        otherwise
            error('Wrong object model!');
    end
    obj.oProbGradAllG.fuseAddTranProbsGPUSimplified(obj.probGradsG .* obj.stepSizeFactor(2), disYs, disXs, weights);
    obj.object_normalizationG = obj.object_normalizationG + sum(abs(obj.midObjWavesG).^2, 3); % (nYX, nYX)
end