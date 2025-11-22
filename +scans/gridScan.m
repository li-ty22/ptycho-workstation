function [scanCoors, stepSize, scanN] = gridScan(scanNYX, scanArea, rotAngle)
    % scanNYX: number of scans along Y/X axis
    % scanArea: [m]
    % rotAngle: [degree] clockwise + / anti-clockwise -, rotation around the center of the scan area
    % scanCoors: % (scanN, 2) 2:[m] (y, x)
    % stepSize: [m]

    scanN = scanNYX * scanNYX;
    scanCoors = zeros(scanN, 2);
    y0 = - scanArea / 2;
    x0 = - scanArea / 2; % origin locates at the center of the scan area
    stepSize = scanArea / (scanNYX - 1);
    rotAngleRad = deg2rad(rotAngle);
    rotm = [cos(rotAngleRad), -sin(rotAngleRad); ...
        sin(rotAngleRad), cos(rotAngleRad)]; % clockwise is +
    for iSY = 1:scanNYX
        for iSX = 1:scanNYX
            ind = (iSY - 1) * scanNYX + iSX; % order: row by row 
            y = y0 + (iSY - 1) * stepSize;
            x = x0 + (iSX - 1) * stepSize;
            r_rot = rotm * [x; y];
            scanCoors(ind, 1) = r_rot(2);
            scanCoors(ind, 2) = r_rot(1);
        end
    end
end