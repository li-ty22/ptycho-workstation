function diffMask = makeBFMask(obj, threshold)
    % please run this function when the cbeds have been already "loadAndPreprocess"ed and "alignment"ed!
    % threshold: like 0.9

    avgCBED = obj.getAvgCBED();
    diffMask = avgCBED > (threshold * max(avgCBED, [], 'all'));

end