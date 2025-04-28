import numpy as np
import random
import math
import queue
import sys
import pandas as pd
import json
import numpy as np
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
import json
import time
import os
from train import reward,DeepQNetwork,train
from reset_step import reset



os.environ["PATH"] += ";C:/Program Files/Graphviz/bin/"



NUM_EPISODE = 10  

n_iot = 4  
n_edge = 2  
n_cloud = 1  
n_time = 2000  
n_task_type = 10
n_no_task = 50
max_delay = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]  
task_size = [8.106834 - 1, 8.106834, 8.106834 + 1, 8.106834 + 2, 8.106834 + 3,
             8.106834 + 4, 8.106834 + 5, 8.106834 + 6, 8.106834 + 7, 8.106834 + 8]
task_cpu_cycle_count = [2.39864428 + 3 - 0.5, 2.39864428 + 3 - 1,
                        2.39864428 + 3 - 2, 2.39864428 + 3 + 0.5,
                        2.39864428 + 3 + 1, 2.39864428 + 3 + 2,
                        2.39864428 + 3 - 0.5, 2.39864428 + 3 - 1,
                        2.39864428 + 3 + 0.5, 2.39864428 + 3 + 1]
keep_alive = [2, 2.5, 1.5, 1, 2.5, 3, 2.5, 4, 1.5, 2]  
comp_cap_iot = 1.5 * np.ones(n_iot)  
comp_cap_edge = 5.8 * np.ones([n_edge])  
comp_cap_cloud = 12.45 * np.ones([n_cloud])  

esc = 1e-27

pull_uplink_rate_iot = 14.000000 / 1.0  
pull_uplink_rate_edge = 24.000000 / 1.0  
pull_uplink_rate_cloud = 50.000000 / 1.0  
tran_edge_cap_iot = 8.000000 * np.ones(
    [n_iot, n_edge])  
tran_cloud_cap_iot = 8.000000 / 1.19 * np.ones(
    [n_iot, n_cloud])  

edge_cps = 0.01 * 5  
cloud_cps = 0.002 * 3  
edge_request = 6e-3  
cloud_request = 2e-2  

tr_power = 0.005  

n_actions = 1 + n_edge + n_cloud

n_features_task = (1 +  
                   1 +  1 +    1 +   1  )
n_features_system_iot = (
            n_task_type +  
            n_task_type +  
            n_task_type +  n_task_type +  n_task_type +  1 +  1 +  1 +  1 +  1 +  1 +  n_edge +  n_cloud  )
n_features_system_edge = (1 +   1 + 1 +  1 +  n_task_type + n_task_type +  n_task_type  )
n_features_system_cloud = (1 +   1 +  1 +   1 +   n_task_type +  n_task_type +   n_task_type   )
n_features_task_results = (1 +   1 +   1 +  1 +  1 +   1 +   1 + 1 +   1)  

n_features_system_other = (1 +    1 +   1)  
n_features = (n_features_task + n_features_system_iot
              + n_features_system_edge * n_edge
              + n_features_system_cloud * n_cloud  # +1
              )

Queue_iot_comp = list()
Queue_iot_tran = list()
Queue_edge_comp = list()
Queue_cloud_comp = list()

for iot in range(n_iot):
    Queue_iot_comp.append(queue.Queue())
    Queue_iot_tran.append(queue.Queue())
    Queue_edge_comp.append(list())
    Queue_cloud_comp.append(list())
    for edge in range(n_edge):
        Queue_edge_comp[iot].append(queue.Queue())
    for cloud in range(n_cloud):
        Queue_cloud_comp[iot].append(queue.Queue())

# 待执行总量
all_task_wait_iot_comp_cpu_cycle = np.zeros(n_iot)
all_task_wait_iot_tran_size = np.zeros(n_iot)
all_task_wait_edge_comp_size_cpu_cycle = np.zeros([n_iot, n_edge])
all_task_wait_cloud_comp_size_cpu_cycle = np.zeros([n_iot, n_cloud])
# 各设备各类型待执行任务的数量，初始时全为0
all_task_wait_iot_comp_count = np.zeros([n_iot, n_task_type])
all_task_wait_iot_tran_count = np.zeros([n_iot, n_task_type])
all_task_wait_iot_tran_edge_count = np.zeros([n_iot, n_edge])
all_task_wait_edge_comp_count = np.zeros([n_iot, n_edge, n_task_type])
all_task_wait_cloud_comp_count = np.zeros([n_iot, n_cloud, n_task_type])

per_task_wait_iot_comp_count_cpu_cycle = np.zeros([n_iot, n_task_type])
per_task_wait_iot_tran_count_size = np.zeros([n_iot, n_task_type])
per_task_wait_edge_comp_count_cpu_cycle = np.zeros([n_iot, n_edge, n_task_type])
per_task_wait_cloud_comp_count_cpu_cycle = np.zeros([n_iot, n_cloud, n_task_type])

# 执行器
task_on_process_local = list()
task_on_transmit_local = list()
task_on_process_edge = list()
task_on_process_cloud = list()
for iot in range(n_iot):
    task_on_process_local.append({'function_id': -1, 'data_size': 0,
                                  'cpu_cycle': 0, 'task_arrive_time': 0,
                                  'flag': 0, 'critical_value': 0, 'max_delay': 0,
                                  'remain': 0})
    task_on_transmit_local.append({'function_id': -1, 'data_size': 0,
                                   'cpu_cycle': 0, 'task_arrive_time': 0,
                                   'edge_logs': 0, 'cloud': 0, 'max_delay': 0,
                                   'remain': 0, 'iot_action': 0})

    task_on_process_edge.append(list())
    task_on_process_cloud.append(list())
    for edge in range(n_edge):
        task_on_process_edge[iot].append({'function_id': -1, 'data_size': 0,
                                          'cpu_cycle': 0, 'task_arrive_time': 0,
                                          'tran_finish_time': 0, 'max_delay': 0,
                                          'flag': 0, 'critical_value': 0, 'remain': 0})
    for cloud in range(n_cloud):
        task_on_process_cloud[iot].append({'function_id': -1, 'data_size': 0,
                                           'cpu_cycle': 0, 'task_arrive_time': 0,
                                           'tran_finish_time': 0, 'max_delay': 0,
                                           'flag': 0, 'critical_value': 0, 'remain': 0})

t_iot_comp = np.zeros([n_iot, n_task_type])  
t_iot_tran = np.zeros([n_iot, n_task_type])
b_edge_comp = np.zeros([n_edge, n_task_type])
b_cloud_comp = np.zeros([n_cloud, n_task_type])

# 记录上次服务器调用功能函数的具体时间
iot_pool = np.zeros([n_iot, n_task_type])  # - np.ones
edge_pool = list()
for edge_index in range(n_edge):
    edge_pool.append(np.zeros([n_iot, n_task_type]))
cloud_pool = list()
for cloud_index in range(n_cloud):
    cloud_pool.append(np.zeros([n_iot, n_task_type]))
# max_value = np.max(test[0][:, column_index])     np.max(edge_pool[edge_index][:, n_task_type])

weight_1 = 1  
weight_2 = 18 
weight_3 = 180  
weight_4 = 0.2  


reward_list_record = []
delay_list_record = []
delay_avg_list_record = []
energy_list_record = []
expense_list_record = []
energy_avg_list_record = []
expense_avg_list_record = []
unfinish_total_list_record = []
unfinish_total_list_record0 = []  
unfinish_total_list_record1 = []  
unfinish_total_list_record2 = []  
unfinish_total_list_record3 = []  
unfinish_total_list_record4 = []  
unfinish_total_list_record5 = []  
utilization_ratio_record = [] 
loss_list_record = []  
action_record_iot = []  
action_record_edge = []
action_record_cloud = []
choose_action_by_random = []
choose_action_by_net = []
slot_bandwidth_record = []
cold_count_list_record = []
warm_count_list_record = []
unfinish_indi_record = []
sucess_count_list_record = []

action = []
choose_action_way = []
reward_list = []
delay_list = []
energy_list = []
expense_list = []
unfinish_indi = []
cold_count_list = []
wait_trans_process_delay = []
transing_process_delay = []
wait_comp_process_delay = []
cold_start_delay = []
computing_process_delay = []
history_observation = []
reward_indicator = []

for i in range(n_iot):
    action.append(-np.ones([n_time, 1]))
    choose_action_way.append(-np.ones([n_time, 1]))
    reward_list.append(np.zeros([n_time, 1]))
    delay_list.append(np.zeros([n_time, 1]))
    energy_list.append(np.zeros([n_time, 1]))
    expense_list.append(np.zeros([n_time, 1]))
    unfinish_indi.append(-np.ones([n_time, 1]))
    cold_count_list.append(-np.ones([n_time, 1]))
    wait_trans_process_delay.append(np.zeros([n_time, 1]))
    transing_process_delay.append(np.zeros([n_time, 1]))
    wait_comp_process_delay.append(np.zeros([n_time, 1]))
    cold_start_delay.append(np.zeros([n_time, 1]))
    computing_process_delay.append(np.zeros([n_time, 1]))
    history_observation.append(np.zeros([n_time, 1], dtype=object))



if __name__ == "__main__":
    task = []
    for i in range(n_iot):
        with open(f'../任务列表/iot_{i}_trace_4_new.json', 'r') as f:
            task.append(json.load(f))

    total_task = 0
    total_task_non = 0
    before_max_delay = 0
    after_max_delay = 0

    for i in range(n_iot):
        for j in range(n_time):
            if j <= (n_time - n_no_task):

                if task[i][j][0]['function_id'] != -1:
                    before_max_delay += task[i][j][0]['max_delay']
                    task[i][j][0]['max_delay'] = task[i][j][0]['max_delay']
                    after_max_delay += task[i][j][0]['max_delay']
                    total_task = total_task + 1
                else:
                    total_task_non = total_task_non + 1
            else:
                task[i][j][0]['function_id'] = -1
                task[i][j][0]['data_size'] = 0
                task[i][j][0]['cpu_cycle'] = 0
                total_task_non = total_task_non + 1

    with open('../任务数txt/record-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.txt',
              'w') as f:
        f.write(f"[{total_task}, {total_task_non},{before_max_delay},{after_max_delay}]\n")


    iot_RL_list = list()  
    for iot in range(n_iot):
        iot_RL_list.append(DeepQNetwork(n_actions, n_features, n_time))

    train(
        iot_RL_list, NUM_EPISODE, n_iot, n_edge, n_cloud, n_task_type, n_time, n_features, keep_alive,

        # 通信与计算能力参数
        pull_uplink_rate_iot, comp_cap_iot,
        tran_edge_cap_iot, tran_cloud_cap_iot,
        pull_uplink_rate_edge, pull_uplink_rate_cloud,
        comp_cap_cloud, comp_cap_edge,

        # 奖励与任务参数
        reward_indicator, task, weight_1, weight_2, weight_3,
        choose_action_way, n_actions, n_no_task,

        # 历史记录与训练统计
        history_observation,
        reward_list_record, delay_list_record, energy_list_record, expense_list_record,
        unfinish_total_list_record, unfinish_total_list_record0, unfinish_total_list_record1,
        unfinish_total_list_record2, unfinish_total_list_record3, unfinish_total_list_record4,
        unfinish_total_list_record5,
        action_record_iot, action_record_edge, action_record_cloud,
        choose_action_by_random, choose_action_by_net,
        cold_count_list_record, warm_count_list_record, sucess_count_list_record,
        delay_list, action, reward_list, energy_list, expense_list, unfinish_indi,
        cold_count_list, cold_start_delay, transing_process_delay, computing_process_delay,

        # IoT 相关队列与统计
        Queue_iot_comp, all_task_wait_iot_comp_count,
        per_task_wait_iot_comp_count_cpu_cycle, all_task_wait_iot_comp_cpu_cycle,
        Queue_iot_tran, all_task_wait_iot_tran_count, all_task_wait_iot_tran_edge_count,
        per_task_wait_iot_tran_count_size, all_task_wait_iot_tran_size,
        task_on_transmit_local, task_on_process_local,

        # Edge 相关
        task_on_process_edge, wait_comp_process_delay, Queue_edge_comp, wait_trans_process_delay,
        esc, all_task_wait_edge_comp_count, per_task_wait_edge_comp_count_cpu_cycle,
        all_task_wait_edge_comp_size_cpu_cycle,

        # Cloud 相关
        Queue_cloud_comp, all_task_wait_cloud_comp_count,
        per_task_wait_cloud_comp_count_cpu_cycle, all_task_wait_cloud_comp_size_cpu_cycle,
        tr_power, task_on_process_cloud
    )

    print('Training Finished')




