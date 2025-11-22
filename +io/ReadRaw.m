function data = ReadRaw(file, nY, nX, nZ, complexFlag)
% The datatype in raw file is float (4 bytes)
% The additional two rows are appended to the end of the data
% file: 'xxx.raw'
% complexFlag: complex data or not
% data: (nY, nX, nZ)
    
    fid = fopen(file, 'rb');
    
    if complexFlag
        data = complex(zeros(nY, nX, nZ, 'single'), zeros(nY, nX, nZ, 'single'));
        for z = 1 : nZ
            rawImgReal = fread(fid, [nX, nY+2], 'single');
            rawImgImag = fread(fid, [nX, nY+2], 'single');
            img = complex(rawImgReal(:, 1:end-2), rawImgImag(:, 1:end-2));
            data(:, :, z) = img.';
        end
    else
        data = zeros(nY, nX, nZ, 'single');
        for z = 1 : nZ
            rawImg = fread(fid, [nX, nY+2], 'single');
            img = rawImg(:, 1:end-2);
            data(:, : ,z) = img.';
        end
    end

    fclose(fid);

end