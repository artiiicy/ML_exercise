import NeuralNetworkclass as NNclass

# hyper-parameter values
batch_size = 60
epoch_num = 10000
learning_rate = 0.003
hidden_size = 15

input_size = 4
output_size = 3

# Two layer neural network 생성 후 학습
twoLayerNetwork = NNclass.TwoLayerNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
twoLayerNetwork.learn(lr=learning_rate, epoch=epoch_num, batch_size=batch_size, verbose=True)
