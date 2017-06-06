#!/usr/bin/env python
# coding: utf-8

import csv

import keras as k
import numpy as np
import tensorflow as tf

batch_size = 2500
epochs = 100000
validation_split = 0.1
lr = 1e-3
seq_size = 20  # 50ms * 20 = 1s
cell_size = 128
h_size = 128
loss = k.losses.mse

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def build_rnn(cell_type=None, isBidirectional=False, isBatchNorm=False, isNoise=False, isDropout=False):
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

    def dropout(layer, isDropout):
        if isDropout:
            return k.layers.GaussianDropout(0.1)(layer)
        else:
            return layer

    input_main_data = k.layers.Input((seq_size, x_main.shape[2]), name='input_main')
    input_aux_data = k.layers.Input((seq_size, x_aux.shape[2]), name='input_aux')

    if isNoise:
        gaussian_input_main_data = k.layers.GaussianNoise(0.1, name='gaussian_main')(input_main_data)
        cell_main = dropout(batchNorm(
            bidirectionalCell(rnn_cell(cell_size, return_sequences=True, name='cell_main_1'), isBidirectional)(
                gaussian_input_main_data), isBatchNorm), isDropout)
    else:
        cell_main = dropout(batchNorm(
            bidirectionalCell(rnn_cell(cell_size, return_sequences=True, name='cell_main_1'), isBidirectional)(
                input_main_data), isBatchNorm), isDropout)

    aux_output = dropout(batchNorm(k.layers.TimeDistributed(k.layers.Dense(1, name='aux_output'))(cell_main), isBatchNorm), isDropout)

    if isNoise:
        gaussian_input_aux_data = k.layers.GaussianNoise(0.1)(input_aux_data)
        cell_aux = dropout(batchNorm(
            bidirectionalCell(rnn_cell(cell_size, return_sequences=True, name='cell_aux_1'), isBidirectional)(
                gaussian_input_aux_data), isBatchNorm), isDropout)
    else:
        cell_aux = dropout(batchNorm(
            bidirectionalCell(rnn_cell(cell_size, return_sequences=True, name='cell_aux_1'), isBidirectional)(
                input_aux_data),
            isBatchNorm), isDropout)

    cell = k.layers.concatenate([cell_main, cell_aux], name='concatenate')
    cell = dropout(batchNorm(
        bidirectionalCell(rnn_cell(cell_size, return_sequences=True, name='encode_layer'), isBidirectional)(cell),
        isBatchNorm), isDropout)
    cell = dropout(batchNorm(
        bidirectionalCell(rnn_cell(cell_size, return_sequences=True, name='encode_layer1'), isBidirectional)(cell),
        isBatchNorm), isDropout)

    flatten = k.layers.Flatten(name='flatten')(cell)
    encode_output = dropout(batchNorm(k.layers.Dense(h_size, name='dnn_all_1', activation='tanh')(flatten), isBatchNorm), False)
    encode_output = k.layers.RepeatVector(seq_size, name='encode_output')(encode_output)

    aux_output1 = dropout(batchNorm(k.layers.TimeDistributed(k.layers.Dense(1, name='aux_output'))(encode_output), isBatchNorm), isDropout)

    cell = dropout(batchNorm(
        bidirectionalCell(rnn_cell(cell_size, return_sequences=True, name='decode_layer_1'), isBidirectional)(
            encode_output),
        isBatchNorm), isDropout)
    cell = dropout(batchNorm(
        bidirectionalCell(rnn_cell(cell_size, return_sequences=True, name='decode_layer_2'), isBidirectional)(cell),
        isBatchNorm), isDropout)
    decode_output = dropout(batchNorm(k.layers.TimeDistributed(k.layers.Dense(1, name='decode_output'))(cell), isBatchNorm), isDropout)

    model = k.models.Model([input_main_data, input_aux_data], [decode_output, aux_output, aux_output1])
    model.summary()

    model.compile(optimizer=k.optimizers.Adam(lr=lr), loss=loss, metrics=[k.metrics.mae], loss_weights=[0.7, 0.1, 0.2])
    return model


def train():
    model = build_rnn(cell_type=None, isBatchNorm=True, isBidirectional=True, isNoise=True, isDropout=True)
    try:
        model.load_weights('./model/bmw/checkpoint/model_128_noise_all2-1799-0.16-0.48.hdf5')
        print('model load complete !!')
    except Exception:
        print('not model!')
    model.fit(x=[x_main[0: (x_main.shape[0] - 1)], x_aux[0: (x_aux.shape[0] - 1)]], y=[y[1: y.shape[0]]] * 3,
              # model.fit(x=[x_main, x_aux], y=[y] * 3,
              batch_size=batch_size, epochs=epochs,
              callbacks=[
                  k.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, verbose=1, cooldown=50),
                  k.callbacks.ModelCheckpoint(
                      filepath='model/bmw/checkpoint/model_128_noise_all3-{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5',
                      verbose=1, period=300, save_best_only=False, mode='min'),
                  k.callbacks.TensorBoard(histogram_freq=50, write_graph=True, write_images=False)],
              validation_split=validation_split, shuffle=False)
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


def pre_data(x):
    x_list = []
    print(x.shape[0], '---------------------')
    for i in range((x.shape[0] - seq_size + 1) / 20):
        x_list.append(x[i * 20: (i * 20 + seq_size)])
    return np.array(x_list)


if __name__ == '__main__':
    # x_main, x_aux, y = getData()
    #
    # np.save('./data/bmw/x_main_all', x_main)
    # np.save('./data/bmw/x_aux_all', x_aux)
    # np.save('./data/bmw/y_all', y)

    from sklearn import preprocessing as pre

    # x_main = np.load('./data/bmw/x_main.npy')
    # x_main = pre.scale(x_main.reshape([-1, 60 * 24]))
    # x_main = x_main.reshape([-1, 60, 24])
    # x_aux = np.load('./data/bmw/x_aux.npy')
    # x_aux = pre.scale(x_aux.reshape([-1, 60 * 15]))
    # x_aux = x_aux.reshape([-1, 60, 15])
    # y = np.load('./data/bmw/y.npy')

    x_main = np.load('./data/bmw/x_main_all.npy')
    x_main = pre.scale(x_main.reshape([-1, 60 * 24]))
    x_main = x_main.reshape([-1, seq_size, 24])
    x_aux = np.load('./data/bmw/x_aux_all.npy')
    x_aux = pre.scale(x_aux.reshape([-1, 60 * 15]))
    x_aux = x_aux.reshape([-1, seq_size, 15])
    y = np.load('./data/bmw/y_all.npy')
    y = y.reshape([-1, 60 * 1]).reshape([-1, seq_size, 1])

    # x_main = np.load('./data/bmw/x_main_all.npy')
    # x_main = x_main.reshape([-1, 24])[:-seq_size]
    # x_main = pre.scale(x_main)
    # x_main = pre_data(x_main)
    # x_aux = np.load('./data/bmw/x_aux_all.npy')
    # x_aux = pre.scale(x_aux.reshape([-1, 15]))[:-seq_size]
    # x_aux = pre_data(x_aux)
    # y = np.load('./data/bmw/y_all.npy').astype(np.float32)
    # y = y.reshape([-1, 1])[seq_size:]
    # y = pre_data(y)

    print(x_main.shape, x_aux.shape, y.shape)
    print(x_main[1][-1])
    print(y[0][0])

    train()
