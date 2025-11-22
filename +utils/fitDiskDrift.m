function [cbed_fit_CoMs] = fitDiskDrift(oCBED, YXInds, fitFunc)
%FITDISKDRIFT 
%   YXInds: (subAreaScanN, 2), 2: (scanY, scanX)
%   fitFunc: 'constant' or 'plane'
%
%   cbed_fit_CoMs: (subAreaScanN, 2), 2: (v, u)
    
    cbed_measured_CoMs = oCBED.calCoMs(); % (subAreaScanN, 2), 2: (v, u)
    switch fitFunc
        case 'constant'
            cbed_fit_CoMs = mean(cbed_measured_CoMs, 1);
        case 'plane'
            p_v = lsqcurvefit(@fit_plane, zeros(1, 3), double(YXInds), double(cbed_measured_CoMs(:, 1)));
            p_u = lsqcurvefit(@fit_plane, zeros(1, 3), double(YXInds), double(cbed_measured_CoMs(:, 2)));
            cbed_fit_CoMs = [fit_plane(p_v, double(YXInds)), fit_plane(p_u, double(YXInds))];
    end

    function F = fit_plane(x, xdata)
        F = x(1) * xdata(:, 1) + x(2) * xdata(:, 2) + x(3);
    end
    
end

