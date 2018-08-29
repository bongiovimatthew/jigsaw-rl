from Brain.dnn import DeepNet, DnnAgent
from Brain.kerasNet import KerasNet


class NetFactory:

    def makeKerasNet(action_space, lr, stateSize):
        return KerasNet(action_space, lr, stateSize)

    def makeNetAgent():
        return DnnAgent

    def makeNet(action_space, lr, stateSize):
        return NetFactory.makeCntkDnn(action_space, lr, stateSize)

    def makeCntkDnn(action_space, lr, stateSize):
        return DeepNet(action_space, lr, stateSize)
