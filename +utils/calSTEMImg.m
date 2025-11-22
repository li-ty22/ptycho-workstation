function stemImg = calSTEMImg(oCBED, mask, scanNY, scanNX)
    if scanNY * scanNX ~= oCBED.scanN
        error('scanNY x scanNX is not equal to scanN in oCBED');
    end
    stemImg = nan(scanNY, scanNX, 'single');
    stemImg1D = reshape(stemImg.', oCBED.scanN, 1);
    stemImg1D(oCBED.subAreaScanIDs) = squeeze(sum(oCBED.cbed .* mask, [1 2]));
    stemImg = reshape(stemImg1D, scanNX, scanNY).'; % cbed's order: row by row
end

