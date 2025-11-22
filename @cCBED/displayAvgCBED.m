function displayAvgCBED(obj, varargin)
% average CBED and the radius of disk
    % varargin: diskRadiusPix log
    p = inputParser;
    addParameter(p, 'diskRadiusPix', obj.calDiskRadiusPix());
    addParameter(p, 'log', false);
    parse(p, varargin{:});
    
    diskRadiusPix = p.Results.diskRadiusPix;
    isLog = p.Results.log;

    avgCBED = obj.getAvgCBED();
    imgN = size(avgCBED, 1);
    center = floor(imgN / 2) + 1;
    figure(), hold on;
    if isLog
        imagesc(log(1+avgCBED)), axis equal, ...
            set(gca, 'XLim', [1, imgN]), set(gca, 'YLim', [1, imgN]), ...
            title(['Log average CBED with diskRadius ', num2str(diskRadiusPix), ' pix'], 'FontSize', 14);
    else
        imagesc(avgCBED), axis equal, ...
            set(gca, 'XLim', [1, imgN]), set(gca, 'YLim', [1, imgN]), ...
            title(['average CBED with diskRadius ', num2str(diskRadiusPix), ' pix'], 'FontSize', 14);
    end
    rectangle('Position', [center-diskRadiusPix, center-diskRadiusPix, 2*diskRadiusPix, 2*diskRadiusPix], ...
        'Curvature', [1,1], 'EdgeColor', 'w', 'LineStyle', '--', 'LineWidth', 2.0);
end