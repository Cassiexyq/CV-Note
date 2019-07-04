# -*- coding: utf-8 -*-

# @Author: xyq


import numpy as np


def inference(theta,x):
    pred_y = sigmod(np.dot(theta, x))
    return pred_y


def sigmod(z):
    return 1 / (1 + np.exp(-z))


def eval_loss(theta,x_list,gt_y_list): # 一个batch 的损失
    avg_loss = 0.0
    num_acc = 0
    for i in range(len(x_list)):
        y_pred = sigmod(np.dot(x_list[i], theta))
        if y_pred >= 0.5 and gt_y_list[i] == 1:
            num_acc += 1
        if y_pred < 0.5 and gt_y_list[i] == 0:
            num_acc += 1
        avg_loss += gt_y_list[i] * np.log(y_pred) \
                    + (1 - gt_y_list[i]) * np.log(1 - y_pred)
    avg_loss /= len(gt_y_list)
    acc = num_acc / len(gt_y_list)
    return -avg_loss, acc


def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dtheta = diff * x
    return dtheta


def cal_step_gradient(batch_x_list, batch_gt_y_list, theta, lr):
    avg_dt = 0
    batch_size = len(batch_x_list)
    for i in range(batch_size):
        pred_y = inference(theta,batch_x_list[i])
        dtheta = gradient(pred_y, batch_gt_y_list[i],batch_x_list[i])
        avg_dt += dtheta
    avg_dt /= batch_size
    theta -= lr * avg_dt
    return theta


def train(x_list,gt_y_list, batch_size, lr, max_iter):
    theta = 0
    for i in range(max_iter):
        batch_idx = np.random.choice(len(x_list), batch_size)
        batch_x = [x_list[j] for j in batch_idx]
        batch_y = [gt_y_list[j] for j in batch_idx]
        theta = cal_step_gradient(batch_x, batch_y, theta, lr)
        print('theta:{0}'.format(theta))
        loss, acc = eval_loss(theta,x_list, gt_y_list)
        print('loss is {0}, acc is {1}'.format(loss, acc))


def get_sample_data():
    from sklearn import datasets
    num_sample = 100
    x_list, y_list = datasets.make_moons(n_samples=num_sample, noise=0.3)
    theta = np.random.randint(-1,1,size=(2,1))
    return x_list,y_list, theta


def run():
    x_list,y_list,theta = get_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_list, y_list, 50, lr, max_iter)


# batch 的话就是每次取w,b就是那一个batch的数据去做梯度了
if __name__ == '__main__':
    run()
