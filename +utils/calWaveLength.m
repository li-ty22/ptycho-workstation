function waveLength = calWaveLength(voltage)
% waveLength: [m]
% voltage: [V]
    import utils.physicalConstant;
    physicalConstant; % load the physical constants
    beta = 1.0 + ECHARGE * voltage / (2.0 * EMASS * CSPEED * CSPEED);
    waveLength = CPLANCK / sqrt(2.0 * EMASS * ECHARGE * voltage * beta);
end

