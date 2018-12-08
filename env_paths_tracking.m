function paths = env_paths_tracking(varargin)
    paths.net_base = './pretrained/';
    paths.video_base = './sequences';
    paths.stats = './data/ILSVRC2015.stats.mat'; 
    paths = vl_argparse(paths, varargin);
end
