%input: video sequence name
%output: tracking results
function bboxes = run_HSSNet(video_name) 
% add path and setting enviroment
    warning off;
    startup;
    paths = env_paths_tracking();
    
% initial the object target's parameters  
    [img_files, pos, target_sz]=load_video_info(paths.video_base,video_name);
    for i=1:length(img_files)
        tracker_params.imgFiles{i,:}=single(imread(img_files{1,i}));
    end
    tracker_params.targetPosition = pos;%[cy cx];
    tracker_params.targetSize = target_sz;%round([h w]);
    
% Call the main tracking function
    [bboxes, ~] = tracker(tracker_params);
end
