function displayDiskDrift(oCBED, scanNY, scanNX, interval)
    trans = - oCBED.calDiskTransPixEachFrame();
    tranAvg = - oCBED.calDiskTransPix();
    trans = permute(reshape(trans - tranAvg, scanNX, scanNY, 2), [2 1 3]); % (scanNY, scanNX, 2)
    [X, Y] = meshgrid(1:scanNX, 1:scanNY);

    figure(), hold on;
    indY = 1:interval:scanNY;
    indX = 1:interval:scanNX;
    quiver(X(indY, indX), Y(indY, indX), trans(indY, indX, 1), trans(indY, indX, 2)), ...
        axis equal, ...
        set(gca, 'YDir', 'reverse'), ...
        title('Disk Drift', 'FontSize', 14);
    figure(), hold on;
    subplot(1, 2, 1), histogram(trans(:, :, 1)), title('Histogram of Drift Y', 'FontSize', 14);
    subplot(1, 2, 2), histogram(trans(:, :, 2)), title('Histogram of Drift X', 'FontSize', 14);
    display(['Std DriftY: ', num2str(std(trans(:, :, 1), 0, 'all'), '%6f'), ' Std DriftX: ', num2str(std(trans(:, :, 2), 0, 'all'), '%6f')]);
end

