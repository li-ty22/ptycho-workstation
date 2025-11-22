function diskRadiusPix = semiAngle2diskRadiusPix(semiAngle, waveLength, pSizeF)
% diskRadiusPix = (1 / waveLength) * tan(semiAngle) / pSizeF
% semiAngle [rad]
% waveLength [m]
% pSizeF [1/m]
    diskRadiusPix = (1 / waveLength) * tan(semiAngle) / pSizeF;
end

