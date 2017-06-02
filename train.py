#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import tensorflow.contrib as tc
import keras as k
import sklearn.preprocessing as pre

batch_size = 800
epochs = 10000
validation_split = 0.1
lr = 1e-2
seq_size = 60  # 50ms * 60 = 3s
loss = k.losses.mse

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def build_rnn(cell_type=None, isBidirectional=False, isBatchNorm=False, isNoise=False):
    if cell_type == 'gru':
        rnn_cell = k.layers.GRU
    else:
        rnn_cell = k.layers.LSTM

    def bidirectionalCell(cell, isBidirectional):
        name = cell.name + '_bidi'
        if isBidirectional:
            return k.layers.Bidirectional(cell, name=name)
        else:
            return cell

    def batchNorm(layer, isBatchNorm):
        if isBatchNorm:
            return k.layers.BatchNormalization()(layer)
        else:
            return layer

    input_main_data = k.layers.Input((seq_size, x_main.shape[2]), name='input_main')
    input_aux_data = k.layers.Input((seq_size, x_aux.shape[2]), name='input_aux')
    if isNoise:
        gaussian_input_main_data = k.layers.GaussianNoise(0.1, name='gaussian_main')(input_main_data)
        cell_main = batchNorm(
            bidirectionalCell(rnn_cell(32, return_sequences=True, name='cell_main_1'), isBidirectional)(
                gaussian_input_main_data), isBatchNorm)
    else:
        cell_main = batchNorm(
            bidirectionalCell(rnn_cell(32, return_sequences=True, name='cell_main_1'), isBidirectional)(
                input_main_data), isBatchNorm)

    aux_output = batchNorm(k.layers.TimeDistributed(k.layers.Dense(1, name='aux_output'))(cell_main), isBatchNorm)

    if isNoise:
        gaussian_input_aux_data = k.layers.GaussianNoise(0.1)(input_aux_data)
        cell_aux = batchNorm(bidirectionalCell(rnn_cell(32, return_sequences=True, name='cell_aux_1'), isBidirectional)(
            gaussian_input_aux_data), isBatchNorm)
    else:
        cell_aux = batchNorm(
            bidirectionalCell(rnn_cell(32, return_sequences=True, name='cell_aux_1'), isBidirectional)(input_aux_data),
            isBatchNorm)

    cell = k.layers.concatenate([cell_main, cell_aux], name='concatenate')
    cell = batchNorm(bidirectionalCell(rnn_cell(32, return_sequences=True, name='encode_layer'), isBidirectional)(cell),
                     isBatchNorm)
    cell = batchNorm(
        bidirectionalCell(rnn_cell(32, return_sequences=True, name='encode_layer1'), isBidirectional)(cell),
        isBatchNorm)

    flatten = k.layers.Flatten(name='flatten')(cell)
    encode_output = batchNorm(k.layers.Dense(128, name='dnn_all_1', activation='tanh')(flatten), isBatchNorm)
    encode_output = k.layers.RepeatVector(seq_size, name='encode_output')(encode_output)

    aux_output1 = batchNorm(k.layers.TimeDistributed(k.layers.Dense(1, name='aux_output'))(encode_output), isBatchNorm)

    cell = batchNorm(
        bidirectionalCell(rnn_cell(32, return_sequences=True, name='decode_layer_1'), isBidirectional)(encode_output),
        isBatchNorm)
    cell = batchNorm(
        bidirectionalCell(rnn_cell(32, return_sequences=True, name='decode_layer_2'), isBidirectional)(cell),
        isBatchNorm)
    decode_output = batchNorm(k.layers.TimeDistributed(k.layers.Dense(1, name='decode_output'))(cell), isBatchNorm)

    model = k.models.Model([input_main_data, input_aux_data], [decode_output, aux_output, aux_output1])
    model.summary()

    model.compile(optimizer=k.optimizers.Adam(lr=lr), loss=loss, metrics=[k.metrics.mae], loss_weights=[1, 0.1, 0.2])
    return model


def train():
    model = build_rnn(cell_type=None, isBatchNorm=True, isBidirectional=True, isNoise=True)
    try:
        model.load_weights('./model/bmw/checkpoint/model_128-3499-0.28.hdf5')
        print('model load complete !!')
    except Exception:
        print('not model!')
    model.fit(x=[x_main[0: (x_main.shape[0] - 1)], x_aux[0: (x_aux.shape[0] - 1)]], y=[y[1: y.shape[0]]] * 3,
              batch_size=batch_size, epochs=epochs,
              callbacks=[
                  k.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1, cooldown=100),
                  k.callbacks.ModelCheckpoint(
                      filepath='model_32_noise-{epoch:02d}-{val_loss:.2f}.hdf5',
                      verbose=1, period=101, save_best_only=True, mode='min'),
                  k.callbacks.TensorBoard(histogram_freq=50, write_graph=True, write_images=False)],
              validation_split=validation_split)
    model.save('./model/bmw/final_model.h5')


def getData():
    # reader = csv.DictReader(open('/home/fang/PycharmProjects/experiment/data/bmw/BMWdata1206n.csv'))
    reader = csv.DictReader(open('/home/fang/PycharmProjects/experiment/data/bmw/BMWdata1231_1.csv'))
    x_main = []
    x_aux = []
    y = []

    data_map = {}

    for row in reader:
        if not '' in list(row.values()):
            timestamp_ = row['Timestamp']
            timestamp_ = float(timestamp_)
            if timestamp_ in data_map.keys():
                data_map[timestamp_].append(row)
            else:
                data_map[timestamp_] = [row]

    data_list = sorted(data_map.items(), key=lambda item: item[0])

    result_datas = []

    for datas in data_list:
        d_list = datas[1]

        sLaneDistanceToEgo = [99.0]
        sVelocityAbs = [99.0]
        sAccelerationAbs = [0.0]
        sYawRelToLane = [0.0]
        dlLaneDistanceToEgo = [99.0]
        dlVelocityAbs = [99.0]
        dlAccelerationAbs = [0.0]
        dlYawRelToLane = [0.0]
        drLaneDistanceToEgo = [99.0]
        drVelocityAbs = [99.0]
        drAccelerationAbs = [0.0]
        drYawRelToLane = [0.0]

        for d_map in d_list:
            if d_map['AgentID'] != '0':
                if d_map['LaneRelNr'] == '0':
                    sLaneDistanceToEgo.append(float(d_map['LaneDistanceToEgo']))
                    sVelocityAbs.append(float(d_map['VelocityAbs']))
                    sAccelerationAbs.append(float(d_map['AccelerationAbs']))
                    sYawRelToLane.append(float(d_map['YawRelToLane']))
                elif d_map['LaneRelNr'] == '1':
                    dlLaneDistanceToEgo.append(float(d_map['LaneDistanceToEgo']))
                    dlVelocityAbs.append(float(d_map['VelocityAbs']))
                    dlAccelerationAbs.append(float(d_map['AccelerationAbs']))
                    dlYawRelToLane.append(float(d_map['YawRelToLane']))
                elif d_map['LaneRelNr'] == '-1':
                    drLaneDistanceToEgo.append(float(d_map['LaneDistanceToEgo']))
                    drVelocityAbs.append(float(d_map['VelocityAbs']))
                    drAccelerationAbs.append(float(d_map['AccelerationAbs']))
                    drYawRelToLane.append(float(d_map['YawRelToLane']))
            elif d_map['AgentID'] == '0':
                egoVehicle = d_map
        sLaneDistanceToEgoMin = min(sLaneDistanceToEgo)
        dlLaneDistanceToEgoMin = min(dlLaneDistanceToEgo)
        drLaneDistanceToEgoMin = min(drLaneDistanceToEgo)

        sindex = sLaneDistanceToEgo.index(sLaneDistanceToEgoMin)
        dlindex = dlLaneDistanceToEgo.index(dlLaneDistanceToEgoMin)
        drindex = drLaneDistanceToEgo.index(drLaneDistanceToEgoMin)

        sVelocityAbsVal = sVelocityAbs[sindex]
        dlVelocityAbsVal = dlVelocityAbs[dlindex]
        drVelocityAbsVal = drVelocityAbs[drindex]
        sAccelerationAbsVal = sAccelerationAbs[sindex]
        dlAccelerationAbsVal = dlAccelerationAbs[dlindex]
        drAccelerationAbsVal = drAccelerationAbs[drindex]
        sYawRelToLaneVal = sYawRelToLane[sindex]
        dlYawRelToLaneVal = dlYawRelToLane[dlindex]
        drYawRelToLaneVal = drYawRelToLane[drindex]

        egoVehicle['sLaneDistanceToEgoMin'] = sLaneDistanceToEgoMin
        egoVehicle['dlLaneDistanceToEgoMin'] = dlLaneDistanceToEgoMin
        egoVehicle['drLaneDistanceToEgoMin'] = drLaneDistanceToEgoMin
        egoVehicle['sVelocityAbsVal'] = sVelocityAbsVal
        egoVehicle['dlVelocityAbsVal'] = dlVelocityAbsVal
        egoVehicle['drVelocityAbsVal'] = drVelocityAbsVal
        egoVehicle['sAccelerationAbsVal'] = sAccelerationAbsVal
        egoVehicle['dlAccelerationAbsVal'] = dlAccelerationAbsVal
        egoVehicle['drAccelerationAbsVal'] = drAccelerationAbsVal
        egoVehicle['sYawRelToLaneVal'] = sYawRelToLaneVal
        egoVehicle['dlYawRelToLaneVal'] = dlYawRelToLaneVal
        egoVehicle['drYawRelToLaneVal'] = drYawRelToLaneVal

        result_datas.append(egoVehicle)

    for dmap in result_datas:
        # x.append(
        #     [dmap['ExistenceProbability'], dmap['IndictatorLeft'], dmap['IndicatorRight'], dmap['TypeProbabilityCar'],
        #      dmap['TypeProbabilityTruck'], dmap['TypeProbabilityBike'],
        #      dmap['TypeProbabilityStationary'], dmap['ObjectWidth'], dmap['ObjectHeight'], dmap['ObjectLength'],
        #      dmap['PositionRelX'], dmap['PositionRelY'], dmap['PositionRelZ'], dmap['PositionAbsX'],
        #      dmap['PositionAbsY'], dmap['PositionAbsZ'], dmap['VelocityAbs'], dmap['VelocityRelX'],
        #      dmap['VelocityRelY'], dmap['VelocityLateral'], dmap['AccelerationAbs'], dmap['AccelerationRel'],
        #      dmap['YawAbs'], dmap['YawRel'], dmap['YawRelToLane'], dmap['LaneWidth'], dmap['LaneDistanceToLeftBorder'],
        #      dmap['LaneDistanceToRightBorder'], dmap['LaneAngleToLeftBorder'], dmap['LaneAngleToRightBorder'],
        #      dmap['LaneDistanceToEgo'], dmap['LaneLeftExists'], dmap['LaneRightExists'], dmap['drYawRelToLaneVal'],
        #      dmap['dlYawRelToLaneVal'], dmap['sYawRelToLaneVal'],
        #      dmap['drAccelerationAbsVal'], dmap['dlAccelerationAbsVal'], dmap['sAccelerationAbsVal'],
        #      dmap['drVelocityAbsVal'], dmap['dlVelocityAbsVal'], dmap['sVelocityAbsVal'],
        #      dmap['drLaneDistanceToEgoMin'],
        #      dmap['dlLaneDistanceToEgoMin'], dmap['sLaneDistanceToEgoMin']])

        x_main.append([dmap['VelocityAbs'], dmap['VelocityLateral'], dmap['AccelerationAbs'],
                       dmap['YawAbs'], dmap['YawRelToLane'], dmap['LaneWidth'],
                       dmap['LaneDistanceToLeftBorder'], dmap['LaneDistanceToRightBorder'],
                       dmap['LaneAngleToLeftBorder'], dmap['LaneAngleToRightBorder'],
                       dmap['LaneLeftExists'], dmap['LaneRightExists'],
                       dmap['drYawRelToLaneVal'], dmap['dlYawRelToLaneVal'], dmap['sYawRelToLaneVal'],
                       dmap['drAccelerationAbsVal'], dmap['dlAccelerationAbsVal'], dmap['sAccelerationAbsVal'],
                       dmap['drVelocityAbsVal'], dmap['dlVelocityAbsVal'], dmap['sVelocityAbsVal'],
                       dmap['drLaneDistanceToEgoMin'], dmap['dlLaneDistanceToEgoMin'], dmap['sLaneDistanceToEgoMin']])

        x_aux.append(
            [dmap['ExistenceProbability'], dmap['IndictatorLeft'], dmap['IndicatorRight'], dmap['TypeProbabilityCar'],
             dmap['TypeProbabilityTruck'], dmap['TypeProbabilityBike'], dmap['TypeProbabilityStationary'],
             dmap['ObjectWidth'],
             dmap['ObjectHeight'], dmap['ObjectLength'], dmap['PositionAbsX'], dmap['PositionAbsY'],
             dmap['PositionAbsZ'],
             dmap['VelocityRelX'], dmap['VelocityRelY']])
        # dmap['PositionRelX'], dmap['PositionRelY'], dmap['PositionRelZ'], dmap['AccelerationRel'], dmap['YawRel'], dmap['LaneDistanceToEgo'],

        y.append(dmap['LaneDistanceToLeftBorder'])

    return np.array(x_main)[0: (len(x_main) - len(x_main) % seq_size)].reshape((-1, seq_size, len(x_main[0]))), \
           np.array(x_aux)[0: (len(x_aux) - len(x_aux) % seq_size)].reshape((-1, seq_size, len(x_aux[0]))), \
           np.array(y)[0: (len(y) - len(y) % seq_size)].reshape((-1, seq_size, 1))


if __name__ == '__main__':
    # x_main, x_aux, y = getData()
    #
    # np.save('./data/bmw/x_main_all', x_main)
    # np.save('./data/bmw/x_aux_all', x_aux)
    # np.save('./data/bmw/y_all', y)

    # x_main = np.load('./data/bmw/x_main_all.npy')
    # x_aux = np.load('./data/bmw/x_aux_all.npy')
    # y = np.load('./data/bmw/y_all.npy')

    from sklearn import preprocessing as pre

    x_main = np.load('x_main_all.npy')
    x_main = pre.scale(x_main.reshape([-1, 60 * 24]))
    x_main = x_main.reshape([-1, 60, 24])
    x_aux = np.load('x_aux_all.npy')
    x_aux = pre.scale(x_aux.reshape([-1, 60 * 15]))
    x_aux = x_aux.reshape([-1, 60, 15])
    y = np.load('y_all.npy')
    # y = pre.scale(y)

    print(x_main.shape)
    print(x_aux.shape)
    print(y.shape)

    print(x_main[0:10])
    print(x_aux[0:10])

    print(x_main[0: (x_main.shape[0] - 1)][2, 21])
    print(x_main[0: (x_main.shape[0] - 1)][2, 22])
    print(x_main[0: (x_main.shape[0] - 1)][2, 23])
    print(y[1: y.shape[0]][1, 21])
    print(y[1: y.shape[0]][1, 22])
    print(y[1: y.shape[0]][1, 23])

    # build_rnn('gru', True, True, True)
    train()
