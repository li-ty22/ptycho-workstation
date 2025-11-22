function calGradRGPU(obj)

    % calculate prob gradient
    obj.probGradsG = obj.midChiWavesG .* ...
        conj(obj.midObjWavesG); % (nYX, nYX, nSpotsParallel)

    % calculate obj gradient
    obj.objGradsCG = obj.midChiWavesG .* conj(obj.midIcdWavesG);
    if isequal(obj.objModel, 'potential')
        obj.objGradsPotG = imag(obj.objGradsCG .* conj(obj.midObjWavesG));
    end
    % objGradsC/PotG (nYX, nYX, nSpotsParallel)
    
end