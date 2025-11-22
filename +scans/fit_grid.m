function [M, theta_deg, sy, sx] = fit_grid(tranGrads, cys, cxs, cysInit, cxsInit)
    % tranGrads: (2, nSpotsParallel) 2:(y, x)
    % cys: (1, nSpotsParallel)
    % cxs: (1, nSpotsParallel)
    
    tranGrads_normalized = reshape(tranGrads ./ sqrt(sum(tranGrads.^2, 1)), 2, []);
    r_pre = zeros(4, numel(cys));
    r_pre(1, :) = cys(:);
    r_pre(2, :) = cxs(:);
    r_pre(3, :) = cysInit(:);
    r_pre(4, :) = cxsInit(:);
    options = optimoptions('lsqcurvefit', 'Algorithm', 'levenberg-marquardt');
    p = lsqcurvefit(@fit_grid, [0.01 0.99 0.99], double(r_pre), double(tranGrads_normalized), [-5/180*pi, 0.9, 0.9], [5/180*pi, 1.1, 1.1], options);
    M = [cos(p(1)), sin(p(1)); -sin(p(1)), cos(p(1))] * [p(2), 0; 0, p(3)]; % for (y, x)
    theta_deg = rad2deg(p(1));
    sy = p(2);
    sx = p(3);

    function F = fit_grid(x, xdata)
        % x: x(1) theta, x(2) sy, x(3) sx
        % xdata: (4, scanN), 1-2: (y, x) current scanCoors, 3-4: (y, x) initial scanCoors
        M = [cos(x(1)), sin(x(1)); -sin(x(1)), cos(x(1))] * [x(2), 0; 0, x(3)]; % relative to initial scanCoors
        F = M * xdata(3:4, :) - xdata(1:2, :);
        F = F ./ sqrt(sum(F.^2, 1));
    end
end