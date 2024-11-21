from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .rnn_egg import RNNEGGAgent
from .rnn_egg_ns import RNNNSEGGAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["rnn_egg"] = RNNEGGAgent
REGISTRY["rnn_egg_ns"] = RNNNSEGGAgent
