function sigmaObj = calSigmaObj(voltage)
% sigmaObj: interaction constant in the transmission function
% voltage: [V]
    import utils.physicalConstant;
    import utils.calWaveLength;
    physicalConstant; % load the physical constants
    beta = sqrt(1 - (EMASS * CSPEED^2 / (ECHARGE * voltage + EMASS * CSPEED^2))^2);
    sigmaObj = 2 * pi / voltage / calWaveLength(voltage) / (1 + sqrt(1 - beta^2));
end

