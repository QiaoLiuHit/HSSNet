function net = make_HSSNet(stream1, stream2, join, final, inputs, output, varargin)
% Constructs a Siamese network of two stream nets joined by the join and
% then followed by the final net.
% The stream and final nets can be simple or DAG.
% They should have one input and one output.
% The two streams should be identical except for parameter values.
% The join net is a DAG with 2 inputs and 1 output.
% The inputs must be named 'in1' and 'in2'.
% The inputs and output parameters are the variable names for the resulting net.
% The input names is a cell-array of 2 strings, the output names is a string.

opts.share_all = true;
% List of params to share, or layers whose params should be shared.
% Ignored if share_all is true.
opts.share_params = [];
opts.share_layers = [];
opts = vl_argparse(opts, varargin) ;

stream_outputs = {'br1_out', 'br2_out'};
stream_outputs_New = {'br1_Wfeat', 'br2_multi_feat'};% added SEnet output
join_output = 'join_out';

% Assume that indices of layers are preserved for share_layers.
if ~isa(stream1, 'dagnn.DagNN')
    stream1 = dagnn.DagNN.fromSimpleNN(stream1);
end
if ~isa(stream2, 'dagnn.DagNN')
    stream2 = dagnn.DagNN.fromSimpleNN(stream2);
end
if ~isempty(final)
    if ~isa(final, 'dagnn.DagNN')
        final = dagnn.DagNN.fromSimpleNN(final);
    end
end
if isempty(final)
    join_output = output;
end

if opts.share_all
    opts.share_params = 1:numel(stream1.params);
else
    % Find all params that belong to share_layers.
    opts.share_params = union(opts.share_params, ...
                              params_of_layers(stream1, opts.share_layers));
end

net = dagnn.DagNN();
add_branches(net, stream1, stream2, inputs, stream_outputs, opts.share_params); %net name changed

% add hierarchical fusion network 
addmultilayers(net);

%add  Spatial-aware network which consists of two sub-networks
%add STNet layer
addSTNet(net);
%add SEnet layer
addSENet(net);

add_join(net, join, {'in1', 'in2'}, stream_outputs_New, join_output)
if ~isempty(final)
    add_final(net, final, join_output, output)
end

end

function add_branches(net, stream1, stream2, inputs, outputs, share_inds)
% share_inds is a list of params to share.

    % Assume that both streams have the same names.
    orig_input = only(stream1.getInputs());
    orig_output = only(stream1.getOutputs());
    % Convert param indices to names.
    share_names = arrayfun(@(l) l.name, stream1.params(share_inds), ...
                           'UniformOutput', false);

    rename_unique1 = @(s) ['br1_', s];
    rename_unique2 = @(s) ['br2_', s];
    rename_common = @(s) ['br_', s];
    rename1 = struct(...
        'layer', rename_unique1, ...
        'var', rename_unique1, ...
        'param', @(s) rename_pred(s, @(x) ismember(x, share_names), ...
                                  rename_common, rename_unique1));
    rename2 = struct(...
        'layer', rename_unique2, ...
        'var', rename_unique2, ...
        'param', @(s) rename_pred(s, @(x) ismember(x, share_names), ...
                                  rename_common, rename_unique2));

    add_dag_to_dag(net, stream1, rename1);
    % Values of shared params will be taken from stream2
    % since add_dag_to_dag over-writes existing parameters.
    add_dag_to_dag(net, stream2, rename2);
    net.renameVar(rename_unique1(orig_input), inputs{1});
    net.renameVar(rename_unique2(orig_input), inputs{2});
    net.renameVar(rename_unique1(orig_output), outputs{1});
    net.renameVar(rename_unique2(orig_output), outputs{2});
end

function r = rename_pred(s, pred, rename_true, rename_false)
    if pred(s)
        r = rename_true(s);
    else
        r = rename_false(s);
    end
end

function add_join(net, join, orig_inputs, inputs, output)
    % assert(numel(join.getInputs()) == 2);
    orig_output = only(join.getOutputs());

    rename_join = @(s) ['join_', s];
    add_dag_to_dag(net, join, rename_join);
    for i = 1:2
        net.renameVar(rename_join(orig_inputs{i}), inputs{i});
    end
    net.renameVar(rename_join(orig_output), output);
end

function add_final(net, final, input, output)
    orig_inputs = final.getInputs();
    orig_outputs = final.getOutputs();
    assert(numel(orig_inputs) == 1);
    assert(numel(orig_outputs) == 1);
    orig_input = orig_inputs{1};
    orig_output = orig_outputs{1};

    rename_final = @(s) ['fin_', s];
    add_dag_to_dag(net, final, rename_final);
    net.renameVar(rename_final(orig_input), input);
    net.renameVar(rename_final(orig_output), output);
end

function param_inds = params_of_layers(net, layer_inds)
    layer_params = arrayfun(@(l) l.params, net.layers(layer_inds), ...
                            'UniformOutput', false);
    param_names = cat(2, {}, layer_params{:});
    param_inds = cellfun(@(s) net.getParamIndex(s), param_names);
    param_inds = unique(param_inds);
end



function net=addmultilayers(net)% add multi layers 
    optss.scale = 1 ;
    optss.weightInitMethod = 'gaussian';

   %add max pooling for br1_conv3 
    br1_pool3 = dagnn.Pooling('method', 'max', 'poolSize', [5 5],'pad', 0, 'stride', 1);
    net.addLayer('br1_pool3', br1_pool3, {'br1_x9'}, {'br1_conv_3feat'});
    net.addLayer('br1_BNconv3', dagnn.BatchNorm('numChannels', 384), {'br1_conv_3feat'}, {'br1_Norm_conv_3feat'},{'br_bn_filters3', 'br_bn_bias3', 'br_bn_moments3'});
    net.params(net.getParamIndex('br_bn_filters3')).value= ones(384, 1, 'single');
    net.params(net.getParamIndex('br_bn_bias3')).value=zeros(384, 1, 'single');
    net.params(net.getParamIndex('br_bn_moments3')).value=zeros(384, 2, 'single');
    
    %add max pooling for br1_conv4
    br1_pool4 = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 0, 'stride', 1);
    net.addLayer('br1_pool4', br1_pool4, {'br1_x12'}, {'br1_conv_4feat'});
    net.addLayer('br1_BNconv4', dagnn.BatchNorm('numChannels', 384), {'br1_conv_4feat'}, {'br1_Norm_conv_4feat'},{'br_bn_filters4', 'br_bn_bias4', 'br_bn_moments4'});
    net.params(net.getParamIndex('br_bn_filters4')).value= ones(384, 1, 'single');
    net.params(net.getParamIndex('br_bn_bias4')).value=zeros(384, 1, 'single');
    net.params(net.getParamIndex('br_bn_moments4')).value=zeros(384, 2, 'single');

     %add BN for br1_conv5
    net.addLayer('br1_BNconv5', dagnn.BatchNorm('numChannels', 256), {'br1_out'}, {'br1_Norm_conv_5feat'},{'br_bn_filters5', 'br_bn_bias5', 'br_bn_moments5'});
    net.params(net.getParamIndex('br_bn_filters5')).value= ones(256, 1, 'single');
    net.params(net.getParamIndex('br_bn_bias5')).value=zeros(256, 1, 'single');
    net.params(net.getParamIndex('br_bn_moments5')).value=zeros(256, 2, 'single');
    
    %add concat layer for br1_conv
    net.addLayer('br1_concat', dagnn.Concat(), {'br1_Norm_conv_3feat','br1_Norm_conv_4feat','br1_Norm_conv_5feat'}, {'br1_concat_multi_feat'});
    net.addLayer('br1_conv6', dagnn.Conv('size',[1 1 1024 256],'pad',0,'stride',1,'hasBias',true), {'br1_concat_multi_feat'}, {'br1_multi_feat'}, {'br_conv6f','br_conv6b'});
    net.params(net.getParamIndex('br_conv6f')).value =init_weight(optss, 1, 1, 1024, 256, 'single');  %--->
    net.params(net.getParamIndex('br_conv6b')).value=zeros(256, 1, 'single');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%branch2

    %add max pooling for br2_conv3 
    br2_pool3 = dagnn.Pooling('method', 'max', 'poolSize', [5 5],'pad', 0, 'stride', 1);
    net.addLayer('br2_pool3', br2_pool3, {'br2_x9'}, {'br2_conv_3feat'});
    net.addLayer('br2_BNconv3', dagnn.BatchNorm('numChannels', 384), {'br2_conv_3feat'}, {'br2_Norm_conv_3feat'},{'br_bn_filters3', 'br_bn_bias3', 'br_bn_moments3'});
    net.params(net.getParamIndex('br_bn_filters3')).value= ones(384, 1, 'single');
    net.params(net.getParamIndex('br_bn_bias3')).value=zeros(384, 1, 'single');
    net.params(net.getParamIndex('br_bn_moments3')).value=zeros(384, 2, 'single');
    
    %add max pooling for br2_conv4
    br2_pool4 = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 0, 'stride', 1);
    net.addLayer('br2_pool4', br2_pool4, {'br2_x12'}, {'br2_conv_4feat'});
    net.addLayer('br2_BNconv4', dagnn.BatchNorm('numChannels', 384), {'br2_conv_4feat'}, {'br2_Norm_conv_4feat'},{'br_bn_filters4', 'br_bn_bias4', 'br_bn_moments4'});
    net.params(net.getParamIndex('br_bn_filters4')).value= ones(384, 1, 'single');
    net.params(net.getParamIndex('br_bn_bias4')).value=zeros(384, 1, 'single');
    net.params(net.getParamIndex('br_bn_moments4')).value=zeros(384, 2, 'single');

     %add BN for br2_conv5
    net.addLayer('br2_BNconv5', dagnn.BatchNorm('numChannels', 256), {'br2_out'}, {'br2_Norm_conv_5feat'},{'br_bn_filters5', 'br_bn_bias5', 'br_bn_moments5'});
    net.params(net.getParamIndex('br_bn_filters5')).value= ones(256, 1, 'single');
    net.params(net.getParamIndex('br_bn_bias5')).value=zeros(256, 1, 'single');
    net.params(net.getParamIndex('br_bn_moments5')).value=zeros(256, 2, 'single');
    
    %add concat layer for br2_conv
    net.addLayer('br2_concat', dagnn.Concat(), {'br2_Norm_conv_3feat','br2_Norm_conv_4feat','br2_Norm_conv_5feat'}, {'br2_concat_multi_feat'});
    net.addLayer('br2_conv6', dagnn.Conv('size',[1 1 1024 256],'pad',0,'stride',1,'hasBias',true), {'br2_concat_multi_feat'}, {'br2_multi_feat'}, {'br_conv6f','br_conv6b'});
    net.params(net.getParamIndex('br_conv6f')).value =init_weight(optss, 1, 1, 1024, 256, 'single');  %--->
    net.params(net.getParamIndex('br_conv6b')).value=zeros(256, 1, 'single');
    %net.addLayer('b_concat_relu', dagnn.ReLU(), {'b_concat_multi_feat1'}, {'b_multi_feat'});

end

function net=addSTNet(net)
channel=256; %input_size=[49 49 256]
%add for br1
  % ************************** localization network ****************************
    br1_l_mp1 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
    net.addLayer('br1_l_mp1', br1_l_mp1, {'br1_multi_feat'}, {'br1_y1'});
    br1_l_cnv1 = dagnn.Conv('size',[5 5 channel 20],'pad',0,'stride',1,'hasBias',true);
    net.addLayer('br1_l_cnv1', br1_l_cnv1, {'br1_y1'}, {'br1_y2'}, {'br1_lc1f','br1_lc1b'});
    net.addLayer('br1_l_re1', dagnn.ReLU(), {'br1_y2'}, {'br1_y3'});
    
    % intiial weights conv1
    f = net.getParamIndex('br1_lc1f') ;
    net.params(f).value=single(randn(5,5,channel,20) /sqrt(1*1*channel))/1e8; 
    net.params(f).learningRate=1;
    net.params(f).weightDecay=1;

    f = net.getParamIndex('br1_lc1b') ;
    net.params(f).value=single(zeros(20,1));
    net.params(f).learningRate=2;
    net.params(f).weightDecay=1;

    br1_l_mp2 = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
    net.addLayer('br1_l_mp2', br1_l_mp2, {'br1_y3'}, {'br1_y4'});
    br1_l_cnv2 = dagnn.Conv('size',[5 5 20 20],'pad',0,'stride',1,'hasBias',true);
    net.addLayer('br1_l_cnv2', br1_l_cnv2, {'br1_y4'}, {'br1_y5'}, {'br1_lc2f','br1_lc2b'});
    net.addLayer('br1_l_re2', dagnn.ReLU(), {'br1_y5'}, {'br1_y6'});
    
     % intiial weights conv2
    f = net.getParamIndex('br1_lc2f') ;
    net.params(f).value=single(randn(5,5,20,20) /sqrt(1*1*20))/1e8; 
    net.params(f).learningRate=1;
    net.params(f).weightDecay=1;

    f = net.getParamIndex('br1_lc2b') ;
    net.params(f).value=single(zeros(20,1));
    net.params(f).learningRate=2;
    net.params(f).weightDecay=1;
    

    br1_l_fc1 = dagnn.Conv('size',[6,6,20,50],'pad',0,'stride',1,'hasBias',true);
    net.addLayer('br1_l_fc1', br1_l_fc1, {'br1_y6'}, {'br1_y7'}, {'br1_lfcf','br1_lfcb'});
    net.addLayer('br1_l_re3', dagnn.ReLU(), {'br1_y7'}, {'br1_y8'});
    
    % intiial weights conv3
    f = net.getParamIndex('br1_lfcf') ;
    net.params(f).value=single(randn(6,6,20,50) /sqrt(1*1*20))/1e8; 
    net.params(f).learningRate=1;
    net.params(f).weightDecay=1;

    f = net.getParamIndex('br1_lfcb') ;
    net.params(f).value=single(zeros(50,1));
    net.params(f).learningRate=2;
    net.params(f).weightDecay=1;
    

    % output affine transforms:
    br1_l_out = dagnn.Conv('size',[1,1,50,6],'pad',0,'stride',1,'hasBias',true);  
    net.addLayer('br1_l_out', br1_l_out, {'br1_y8'}, {'br1_aff'}, {'br1_lof','br1_lob'});
    
    % intiial weights conv4
    f = net.getParamIndex('br1_lof') ;
    net.params(f).value=single(randn(1,1,50,6) /sqrt(1*1*50))/1e8; 
    net.params(f).learningRate=1;
    net.params(f).weightDecay=1;

    f = net.getParamIndex('br1_lob') ;
    net.params(f).value=single(zeros(6,1));
    net.params(f).learningRate=2;
    net.params(f).weightDecay=1;
    
    %***** NEED TO SET THE PARAMETERS OF THIS LAST LAYER TO OUTPUT IDENTITY *******

    % ************************** spatial transformer ******************************
    br1_aff_grid = dagnn.AffineGridGenerator('Ho',49,'Wo',49);
    net.addLayer('br1_aff', br1_aff_grid,{'br1_aff'},{'br1_grid'});

    br1_sampler = dagnn.BilinearSampler();
    net.addLayer('br1_samp',br1_sampler,{'br1_multi_feat','br1_grid'},{'br1_ST_feat'});
    % *****************************************************************************
    
    % VERY IMPORTANT: bias the transformation to IDENTITY:
    br1_f_prev = net.params(net.getParamIndex('br1_lof')).value;
    net.params(net.getParamIndex('br1_lof')).value = 0*br1_f_prev;

    br1_b_prev = 0*net.params(net.getParamIndex('br1_lob')).value;
    br1_b_prev(1) = 1; br1_b_prev(4) = 1;
    net.params(net.getParamIndex('br1_lob')).value = br1_b_prev;
    
  
end

function net=addSENet(net)
channel=256; r=16;

%add SENet for branch1----------------------------------
net.addLayer('br1_globalpooling1', dagnn.GlobalPooling(),'br1_ST_feat','br1_SE_pool');

net.addLayer('br1_conv11', dagnn.Conv('size', [1,1,channel,channel/r],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br1_SE_pool', 'br1_conv_11', {'br1_conv11_f', 'br1_conv11_b'});

f = net.getParamIndex('br1_conv11_f') ;
net.params(f).value=single(randn(1,1,channel,channel/r) /sqrt(1*1*channel))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;

f = net.getParamIndex('br1_conv11_b') ;
net.params(f).value=single(zeros(channel/r,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=1;

net.addLayer('br1_SE_relu1', dagnn.ReLU(),'br1_conv_11','br1_SE_relu_1');

net.addLayer('br1_conv12', dagnn.Conv('size', [1,1,channel/r,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br1_SE_relu_1', 'br1_conv_12', {'br1_conv12_f', 'br1_conv12_b'});

f = net.getParamIndex('br1_conv12_f') ;
net.params(f).value=single(randn(1,1,channel/r,channel) /sqrt(1*1*channel/r))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;

f = net.getParamIndex('br1_conv12_b') ;
net.params(f).value=single(zeros(channel,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=1;

net.addLayer('br1_sigmoid1', dagnn.Sigmoid(),'br1_conv_12','br1_sigmoid_1');
net.addLayer('br1_scale1',dagnn.Scale('hasBias',0),{'br1_ST_feat','br1_sigmoid_1'},'br1_Wfeat')

end




