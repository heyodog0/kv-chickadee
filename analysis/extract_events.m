function extract_events()
% Faithful event extraction using the AUTHORS' parseCacheActions (Chettih et al. 2024
% action parsing code). Produces per-event population "barcode" vectors for the
% authoritative event windows/types, per session, for Python analysis.
%
% Event types (parseCacheActions order): 1 visits(no-int), 2 short checks,
% 3 long interactions, 4 caches, 5 retrievals.
% Population vector = mean normSpks (selectAndNormSpikes, useMUA=true) over eventFrames.
% Also saves cacheNum for caches & retrievals (recomputed via the same intVisMatch
% criterion parseSiteInteractions uses) so a retrieval can be matched to its own cache.

base   = '/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/Grid Caching Data';
apc    = '/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/apc_patched';
outdir = '/n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/events';
addpath(apc); addpath(base);
if ~exist(outdir,'dir'); mkdir(outdir); end

dd = dir(base); sess = {};
for i=1:numel(dd)
    if dd(i).isdir && ~isempty(regexp(dd(i).name,'^[A-Za-z]+\d+_\d{6}_\d{6}$','once')) ...
            && exist(fullfile(base,dd(i).name,'alignedSpikesAndPosture.mat'),'file')
        sess{end+1} = dd(i).name; %#ok<AGROW>
    end
end
fprintf('sessions found: %d\n', numel(sess));

for si=1:numel(sess)
    S = sess{si};
    try
        A = load(fullfile(base,S,'alignedSpikesAndPosture.mat')); aligned = A.alignedData;
        Bs = load(fullfile(base,S,'annotatedSeeds.mat')); annotatedSeeds = Bs.annotatedSeeds;
        [normSpks, goodNeur] = selectAndNormSpikes(aligned, true);   % nBins x nGood
        nBins = size(normSpks,1); nGood = numel(goodNeur);
        [eventFrames,eventID,eventTimes,eventLoc] = parseCacheActions(aligned.smPts, annotatedSeeds);

        % cacheNum linkage (same criterion as parseSiteInteractions: intVisMatch)
        countData = annotatedSeeds.countData;
        perchID = nan(numel(countData.newSite),1);
        for nInt=1:numel(countData.newSite)
            tv = find(countData.newPerch<=countData.endSite(nInt),1,'last');
            if ~isempty(tv); perchID(nInt)=countData.perchNum(tv); end
        end
        intVisMatch = perchID==countData.siteNum;
        cacheIdx = find(any(annotatedSeeds.seedChanges>0,2) & intVisMatch);
        retIdx   = find(any(annotatedSeeds.seedChanges<0,2) & intVisMatch);
        cnum_cache = annotatedSeeds.cacheNum(cacheIdx);
        cnum_ret   = annotatedSeeds.cacheNum(retIdx);

        visV   = build_vecs(eventFrames{1}, normSpks, nBins, nGood);
        chkV   = build_vecs(eventFrames{2}, normSpks, nBins, nGood);
        [cacheV,cacheV1,cacheV2] = build_split(eventFrames{4}, normSpks, nBins, nGood);
        [retV,retV1,retV2]       = build_split(eventFrames{5}, normSpks, nBins, nGood);
        visSite=col(eventID{1}); chkSite=col(eventID{2}); cacheSite=col(eventID{4}); retSite=col(eventID{5});
        visLoc=eventLoc{1}; chkLoc=eventLoc{2}; cacheLoc=eventLoc{4}; retLoc=eventLoc{5};
        visT=eventTimes{1}; chkT=eventTimes{2}; cacheT=eventTimes{4}; retT=eventTimes{5};

        save(fullfile(outdir,[S '.mat']), ...
            'visV','visSite','visLoc','visT','chkV','chkSite','chkLoc','chkT', ...
            'cacheV','cacheV1','cacheV2','cacheSite','cacheLoc','cacheT', ...
            'retV','retV1','retV2','retSite','retLoc','retT', ...
            'cnum_cache','cnum_ret','goodNeur','-v7');
        fprintf('[%2d/%d] %-22s vis=%d chk=%d cache=%d retr=%d good=%d\n', ...
            si, numel(sess), S, size(visV,1), size(chkV,1), size(cacheV,1), size(retV,1), nGood);
    catch ME
        fprintf('[%2d/%d] %-22s ERROR: %s\n', si, numel(sess), S, ME.message);
    end
end
end

function V = build_vecs(fr_cell, normSpks, nBins, nGood)
n = numel(fr_cell); V = zeros(n, nGood);
for e=1:n
    f = fr_cell{e}; f = f(f>=1 & f<=nBins);
    if isempty(f); V(e,:)=NaN; else; V(e,:)=mean(normSpks(f,:),1); end
end
end

function [V,V1,V2] = build_split(fr_cell, normSpks, nBins, nGood)
% full-window mean plus first-half / second-half means (measurement-ceiling control)
n = numel(fr_cell); V = zeros(n,nGood); V1 = zeros(n,nGood); V2 = zeros(n,nGood);
for e=1:n
    f = fr_cell{e}; f = f(f>=1 & f<=nBins);
    if isempty(f); V(e,:)=NaN; V1(e,:)=NaN; V2(e,:)=NaN; continue; end
    V(e,:) = mean(normSpks(f,:),1);
    m = floor(numel(f)/2);
    if m>=1 && (numel(f)-m)>=1
        V1(e,:) = mean(normSpks(f(1:m),:),1);
        V2(e,:) = mean(normSpks(f(m+1:end),:),1);
    else
        V1(e,:) = V(e,:); V2(e,:) = V(e,:);
    end
end
end

function c = col(x); c = x(:); end
