FROM nvcr.io/nvidia/pytorch:22.05-py3

RUN chmod 1777 /tmp
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install transformers==4.31.0
RUN pip install openai==1.10.0 
RUN pip install accelerate==0.29.0 
RUN pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
RUN pip install flash-attn==2.5.6
RUN pip install lyft_dataset_sdk nuscenes-devkit plyfile scikit-image==0.19.3 openlanev2==2.1.0 peft fvcore sentencepiece
RUN git clone https://github.com/open-mmlab/mmdetection3d.git -b v1.0.0rc5
RUN cd mmdetection3d/ && python setup.py install
RUN pip install mmdet==2.28.2 mmsegmentation==0.30.0
RUN pip install "opencv-python-headless<4.3"
RUN pip install shapely==1.7.1
RUN pip install onnx==1.16.2 onnxsim==0.4.36
RUN pip install pycuda==2024.1.2
RUN pip install numpy==1.23
RUN pip install onnx_graphsurgeon
