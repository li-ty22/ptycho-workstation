function videoCBED(obj, fileVideo, IDs)
% IDs: (nSpots, 1)
    nSpots = length(IDs);
    
    fig = figure('Name', 'CBED Video', 'Visible', 'off');
    imgN = size(obj.cbed, 1);
    im = cell(nSpots, 1);
    cmap = visual.blackbody_map(1024);
    for i = 1:nSpots
        ind = IDs(i);
        imagesc(log(1+obj.cbed(:, :, ind))), axis equal, axis off, ...
            set(gca, 'XLim', [1, imgN]), set(gca, 'YLim', [1, imgN]), ...
            title(num2str(ind), 'FontSize', 14);
            colormap(cmap);
        drawnow
        frame = getframe(fig);
        im{i} = frame2im(frame);
        disp(i);
    end
    close;

    for idx = 1:nSpots
        [A, map] = rgb2ind(im{idx}, 256);
        if idx == 1
            imwrite(A, map, fileVideo, 'gif', 'LoopCount', Inf, 'DelayTime', 0.2);
        else
            imwrite(A, map, fileVideo, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2);
        end
        disp(idx);
    end
end

