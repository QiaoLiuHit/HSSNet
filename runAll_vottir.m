%run the tracker on all videos in the "sequences" folder
function runAll_vottir
seqTir={
    'birds'
    'horse'
};
    for s=1:numel(seqTir)
        run_HSSNet(seqTir{s});
    end
end
