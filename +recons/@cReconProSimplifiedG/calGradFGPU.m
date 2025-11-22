function calGradFGPU(obj, cbeds1BeatG)
% cbeds1BeatG (nYX, nYX, nSpotsParallel) gpuArray

    % amplitude log-likelihood
    obj.midChiWavesG = ifft2((sqrt(cbeds1BeatG) .* exp(1i .* angle(obj.midDiffsG)) - obj.midDiffsG) .* (obj.maskGradFG .* obj.phaseInvModG));

end