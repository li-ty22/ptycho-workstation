function semiAngle = diskRadiusPix2semiAngle(diskRadiusPix, waveLength, pSizeF)
% semiAngle = atan(diskRadiusPix * pSizeF * waveLength)
% semiAngle [rad]
% waveLength [m]
% pSizeF [1/m]
    semiAngle = atan(diskRadiusPix * pSizeF * waveLength);
end

