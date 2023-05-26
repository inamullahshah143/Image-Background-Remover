import PIL.Image
import uuid

from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator
from flask import Flask, send_file, request

from pathlib import Path

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    data = request.files['file']
    result = process_function(data)
    return result

seg_net = TracerUniversalB7(device='cpu',batch_size=1)

fba = FBAMatting(device='cpu',input_tensor_size=2048,batch_size=1)

trimap = TrimapGenerator()

preprocessing = PreprocessingStub()

postprocessing = MattingMethod(matting_module=fba,trimap_generator=trimap,device='cpu')

interface = Interface(pre_pipe=preprocessing,post_pipe=postprocessing,seg_pipe=seg_net)

def process_function(data):
    image = PIL.Image.open(data)
    bg = interface([image])[0]
    unique_filename = str(uuid.uuid4())
    bg.save('static/results/'+unique_filename+'.png')
    base_url = request.base_url.rsplit('/', 1)[0]
    return {'result':base_url+'/static/results/'+unique_filename+'.png' }

if __name__ == '__main__':
    app.run()


                   