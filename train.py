#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import tensorflow.contrib as tc
import keras as k
import sklearn.preprocessing as pre


batch_size = 100
epochs = 10000
validation_split = 0.1
lr = 1e-3
seq_size = 60  # 50ms * 60 = 3s
loss = k.losses.mae


def build_rnn(cell_type=None, isBidirectional=False, isBatchNorm=False, isNoise=False):
    if cell_type == 'gru':
        rnn_cell = k.layers.GRU
    else:
        rnn_cell = k.layers.LSTM

    def bidirectionalCell(cell, isBidirectional):
        if isBidirectional:
            return k.layers.Bidirectional(cell)
        else:
            return cell

    def batchNorm(layer, isBatchNorm):
        if isBatchNorm:
            return k.layers.BatchNormalization()(layer)
        else:
            return layer

    input_data = k.layers.Input((seq_size, x.shape[1]))
    if isNoise:
        gaussian_input_data = k.layers.GaussianNoise(0.1)(input_data)
        cell = batchNorm(bidirectionalCell(rnn_cell(128, return_sequences=True), isBidirectional)(gaussian_input_data), isBatchNorm)
    else:
        cell = batchNorm(bidirectionalCell(rnn_cell(128, return_sequences=True), isBidirectional)(input_data), isBatchNorm)
    cell = batchNorm(bidirectionalCell(rnn_cell(128, return_sequences=True), isBidirectional)(cell), isBatchNorm)

    flatten = k.layers.Flatten()(cell)
    encode_output = batchNorm(k.layers.Dense(512)(flatten), isBatchNorm)
    encode_output = k.layers.RepeatVector(seq_size)(encode_output)

    cell = batchNorm(bidirectionalCell(rnn_cell(128, return_sequences=True), isBidirectional)(encode_output), isBatchNorm)
    cell = batchNorm(bidirectionalCell(rnn_cell(128, return_sequences=True), isBidirectional)(cell), isBatchNorm)
    decode_output = k.layers.TimeDistributed(k.layers.Dense(1))(cell)

    model = k.models.Model(input_data, decode_output)
    model.summary()

    model.compile(optimizer=k.optimizers.Nadam(lr=lr), loss=loss, metrics=[k.metrics.mae])
    return model


def getData():
    reader = csv.DictReader(open('C:/Users/fangsw/Desktop/BMW/BMWdata1206n.csv'))
    x = []
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
        x.append(
            [dmap['ExistenceProbability'], dmap['IndictatorLeft'], dmap['IndicatorRight'], dmap['TypeProbabilityCar'],
             dmap['TypeProbabilityTruck'], dmap['TypeProbabilityBike'],
             dmap['TypeProbabilityStationary'], dmap['ObjectWidth'], dmap['ObjectHeight'], dmap['ObjectLength'],
             dmap['PositionRelX'], dmap['PositionRelY'], dmap['PositionRelZ'], dmap['PositionAbsX'],
             dmap['PositionAbsY'], dmap['PositionAbsZ'], dmap['VelocityAbs'], dmap['VelocityRelX'],
             dmap['VelocityRelY'], dmap['VelocityLateral'], dmap['AccelerationAbs'], dmap['AccelerationRel'],
             dmap['YawAbs'], dmap['YawRel'], dmap['YawRelToLane'], dmap['LaneWidth'], dmap['LaneDistanceToLeftBorder'],
             dmap['LaneDistanceToRightBorder'], dmap['LaneAngleToLeftBorder'], dmap['LaneAngleToRightBorder'],
             dmap['LaneDistanceToEgo'], dmap['LaneLeftExists'], dmap['LaneRightExists'], dmap['drYawRelToLaneVal'],
             dmap['dlYawRelToLaneVal'], dmap['sYawRelToLaneVal'],
             dmap['drAccelerationAbsVal'], dmap['dlAccelerationAbsVal'], dmap['sAccelerationAbsVal'],
             dmap['drVelocityAbsVal'], dmap['dlVelocityAbsVal'], dmap['sVelocityAbsVal'],
             dmap['drLaneDistanceToEgoMin'],
             dmap['dlLaneDistanceToEgoMin'], dmap['sLaneDistanceToEgoMin']])
        y.append(dmap['LaneDistanceToLeftBorder'])

    return np.array(x)[0: (len(x) - len(x) % seq_size)].reshape((-1, seq_size, 45)), np.array(y)[0: (len(y) - len(y) % seq_size)].reshape((-1, seq_size, 1))


x, y = getData()
print(x.shape)
print(y.shape)


print(x[0: (x.shape[0] - 1)][2, 1])
print(x[0: (x.shape[0] - 1)][2, 2])
print(x[0: (x.shape[0] - 1)][2, 3])
print(y[1: y.shape[0]][1, 1])
print(y[1: y.shape[0]][1, 2])
print(y[1: y.shape[0]][1, 3])


def train():
    model = build_rnn(cell_type='gru', isBatchNorm=True, isBidirectional=True, isNoise=True)
    model.fit(x=x[0: (x.shape[0] - 1)], y=y[1: y.shape[0]], batch_size=batch_size, epochs=epochs,
              callbacks=[], validation_split=validation_split)
