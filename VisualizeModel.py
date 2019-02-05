from keras.utils import plot_model
from keras.models import Model,load_model
import pydot
model = load_model("ResNet50_12.h5")
plot_model(model, to_file='model.png')
