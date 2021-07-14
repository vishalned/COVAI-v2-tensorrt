import engine as eng
from onnx import ModelProto
import tensorrt as trt


engine_name = 'model.plan'
onnx_path = "model.onnx"
batch_size = 1

model = ModelProto()
with open(onnx_path, "rb") as f:
  model.ParseFromString(f.read())

print('-------------------------------')
print('model.graph.input shape ',model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
print('model.graph.input shape ',model.graph.input[0].type.tensor_type.shape.dim[2].dim_value)
print('model.graph.input shape ',model.graph.input[0].type.tensor_type.shape.dim[3].dim_value)
print('-------------------------------')

d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
shape = [ batch_size , 160, 160 ,80, 1]

engine = eng.build_engine(onnx_path, shape= shape)
print(engine)
eng.save_engine(engine, engine_name) 
