# http://francescopochetti.com/fast-neural-style-transfer-sagemaker-deployment/
import os, sys
import json
import numpy as np
import tarfile
import random
import torch

# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)

import data_feeder as df
from har_model import load_model

JSON_CONTENT_TYPE = 'application/json'
NPY_CONTENT_TYPE = 'application/x-npy'

SAVED_MODEL = 'har_model' ## leave off *.pth extension (DON'T name it model.pth !!!!!) # har_model_smooth
ENTRY_POINT = 'har_model.py'

# role = 'arn:aws:iam::821179856091:role/dsv_sage_exec'# revibe AWS acct
role = 'arn:aws:iam::257759225263:role/service-role/AmazonSageMaker-ExecutionRole-20180525T100991'# new AWS acct

def get_data(gpu_preproc=True):
    data_path = '/home/david/data/revibe/boris/npy/har1'
    data_path = '/home/david/data/revibe/boris/npy/test2'
    npy_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
#     random.shuffle(npy_files)
    
    seed = 1234
    cfg = df.Config(seed=seed)
    cfg.window_size = 32
    cfg.test_step = 8
    cfg.batch_size = 50
    cfg.nz = 1
    
    cfg.gpu_preproc = gpu_preproc
    cfg.feats_raw = True
    cfg.feats_fft = True
    cfg.add_mag = True
        
    cfg.labels = ['', '.fidget.', '.walk.', '.run.']
    cfg.label_fxn = lambda s: np.nonzero([int(l in s) for l in cfg.labels])[0][-1]

    sb = df.SigBatcher(npy_files, cfg, train=False)
    
    for b in sb.batch_stream():
        x = np.transpose(b.X, (0,2,1)).astype(np.float32)
        y = b.y
        break
    
    return x,y,sb

def make_payload(x):  
    bs = x.shape[0]
    data = (
        x.flatten().tolist()
    )
    payload = {"bs": bs, "data": data}
    return payload
        
def dump_payload(x, fn='har_payload.json'):   
    payload = make_payload(x)
    with open(fn, 'w') as file:
        json.dump(payload, file)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == "__main__":
    gpu_preproc = False
    
    ## get sample data...
    x,_,sb = get_data(gpu_preproc=gpu_preproc)
    dump_payload(x)
    
    saved_model_file = '{}.pth'.format(SAVED_MODEL)
    onnx_file = '{}.onnx'.format(SAVED_MODEL)
    
    ## load saved model...
    model = load_model('.', saved_model_file, gpu_preproc=gpu_preproc)
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    x_t = torch.from_numpy(x).float().to(device)
    torch_out = model(x_t)
    
    out = to_numpy(torch_out)
    y = np.argmax(out, 1)
    print(y)
    
    # Export the model...
    torch.onnx.export(model,                     # model being run
                      x_t,                         # model input (or a tuple for multiple inputs)
                      onnx_file,                 # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}},
                      verbose=True,
#                       opset_version=10,          # the ONNX version to export the model to
                      )
    
    ## load model and check...
    import onnx
    onnx_model = onnx.load(onnx_file)
#     onnx.checker.check_model(onnx_model)
    
    ## run model and compare outputs...
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_file)
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = np.array(ort_outs)
    print(ort_outs.shape)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print('passed!')
    
    