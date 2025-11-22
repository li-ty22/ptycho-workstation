function DispAndLog(logFileID, logLine)
    disp(logLine);
    fprintf(logFileID, [logLine, '\n']);
end

