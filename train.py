#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import tensorflow.contrib as tc
import keras as k
import sklearn.preprocessing as pre


def getData():
    reader = csv.DictReader(open('/home/fang/PycharmProjects/experiment/BMWdata1206n.csv'))
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

    return np.array(x), np.array(y)


x, y = getData()
print(x.shape[1])
print(y.shape)

batch_size = 100
epochs = 10000
validation_split = 0.2


def build_lstm():
    input_data = k.layers.Input((60, x.shape[1]))
    lstm_cell = k.layers.LSTM(128, return_sequences=True)(input_data)
    lstm_cell = k.layers.LSTM(128, return_sequences=True)(lstm_cell)
    flatten = k.layers.Flatten()(lstm_cell)
    encode_output = k.layers.Dense(512)(flatten)
    model = k.models.Model(input_data, output_data)
    model.summary()
    model.compile(optimizer=k.optimizers.Adam(lr=1e-3), loss=k.losses.binary_crossentropy, metrics=[k.metrics.mae])
    return model


build_lstm()


def train():
    model = build_lstm()
    model.fit(x=None, y=None, batch_size=batch_size, epochs=epochs, callbacks=[], validation_split=validation_split)
