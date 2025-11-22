function WriteRaw(data, file, complexFlag)
% The datatype in raw file is float (4 bytes)
% The additional two rows are appended to the end of the data
% data: image (y, x) or volume (y, x, z)
% file: 'xxx.raw'
% complexFlag: complex data or not
    
    fid = fopen(file, 'wb');
    
    data = single(permute(data, [2 1 3])); % (x, y, z)
    nX = size(data, 1);
    nY = size(data, 2);
    nZ = size(data, 3);
    
    rawImg = zeros(nX, nY+2, 'single');
    if complexFlag
        for z = 1:nZ
            rawImg(:, 1:end-2) = real(data(:, :, z));
            fwrite(fid, rawImg, 'single');
            rawImg(:, 1:end-2) = imag(data(:, :, z));
            fwrite(fid, rawImg, 'single');
        end
    else
        for z = 1:nZ
            rawImg(:, 1:end-2) = data(:, :, z);
            fwrite(fid, rawImg, 'single');
        end
    end

    fclose(fid);

end