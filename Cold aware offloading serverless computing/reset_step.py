import numpy as np
import queue

def reset(n_iot, n_edge,n_cloud,n_task_type,n_time,n_features,keep_alive,pull_uplink_rate_iot,comp_cap_iot,
          tran_edge_cap_iot,tran_cloud_cap_iot,pull_uplink_rate_edge,pull_uplink_rate_cloud,comp_cap_cloud,reward_indicator):
    t_iot_comp = np.zeros([n_iot, n_task_type])
    t_iot_tran = np.zeros([n_iot, n_task_type])
    b_edge_comp = np.zeros([n_edge, n_task_type])
    b_cloud_comp = np.zeros([n_cloud, n_task_type])
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
    for i in range(n_iot):
        action.append(-np.ones([n_time, 1]))
        choose_action_way.append(-np.ones([n_time, 1]))
        reward_list.append(np.zeros([n_time, 1]))
        delay_list.append(np.zeros([n_time, 1]))
        energy_list.append(np.zeros([n_time, 1]))
        expense_list.append(np.zeros([n_time, 1]))  #
        unfinish_indi.append(-np.ones([n_time, 1]))
        cold_count_list.append(-np.ones([n_time, 1]))
        wait_trans_process_delay.append(np.zeros([n_time, 1]))
        transing_process_delay.append(np.zeros([n_time, 1]))
        wait_comp_process_delay.append(np.zeros([n_time, 1]))
        cold_start_delay.append(np.zeros([n_time, 1]))
        history_observation.append(np.zeros([n_time, 1], dtype=object))
        reward_indicator.append(np.zeros([n_time, 1]))
        computing_process_delay.append(np.zeros([n_time, 1]))
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

    all_task_wait_iot_comp_cpu_cycle = np.zeros(n_iot)
    all_task_wait_iot_tran_size = np.zeros(n_iot)
    all_task_wait_edge_comp_size_cpu_cycle = np.zeros([n_iot, n_edge])
    all_task_wait_cloud_comp_size_cpu_cycle = np.zeros([n_iot, n_cloud])
    all_task_wait_iot_comp_count = np.zeros([n_iot, n_task_type])
    all_task_wait_iot_tran_count = np.zeros([n_iot, n_task_type])
    all_task_wait_iot_tran_edge_count = np.zeros([n_iot, n_edge])
    all_task_wait_edge_comp_count = np.zeros([n_iot, n_edge, n_task_type])
    all_task_wait_cloud_comp_count = np.zeros([n_iot, n_cloud, n_task_type])

    per_task_wait_iot_comp_count_cpu_cycle = np.zeros([n_iot, n_task_type])
    per_task_wait_iot_tran_count_size = np.zeros([n_iot, n_task_type])
    per_task_wait_edge_comp_count_cpu_cycle = np.zeros([n_iot, n_edge, n_task_type])
    per_task_wait_cloud_comp_count_cpu_cycle = np.zeros([n_iot, n_cloud, n_task_type])

    task_on_process_local = list()
    task_on_transmit_local = list()
    task_on_process_edge = list()
    task_on_process_cloud = list()
    for iot in range(n_iot):
        task_on_process_local.append({'function_id': -1, 'data_size': 0,
                                      'cpu_cycle': 0, 'task_arrive_time': 0,
                                      'flag': 0, 'critical_value': 0, 'block': 0,
                                      'remain': 0, 'max_delay': 0})
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

    global iot_pool, edge_pool, cloud_pool
    iot_pool = np.zeros([n_iot, n_task_type])  # - np.ones
    edge_pool = list()
    for edge_index in range(n_edge):
        edge_pool.append(np.zeros([n_iot, n_task_type]))
    cloud_pool = list()
    for cloud_index in range(n_cloud):
        cloud_pool.append(np.zeros([n_iot, n_task_type]))
        observation_all = np.zeros([n_iot, n_features - 5])
    for iot_index in range(n_iot):
        for n_task_type_index in range(n_task_type):
            observation_all[iot_index, n_task_type_index] = all_task_wait_iot_comp_count[iot_index][n_task_type_index]
            observation_all[iot_index, n_task_type * 1 + n_task_type_index] = \
            per_task_wait_iot_comp_count_cpu_cycle[iot_index][n_task_type_index]
            observation_all[iot_index, n_task_type * 2 + n_task_type_index] = all_task_wait_iot_tran_count[iot_index][
                n_task_type_index]
            observation_all[iot_index, n_task_type * 3 + n_task_type_index] = \
            per_task_wait_iot_tran_count_size[iot_index][n_task_type_index]
            observation_all[iot_index, n_task_type * 4 + n_task_type_index] = keep_alive[
                n_task_type_index]  
        observation_all[iot_index, n_task_type * 5] = all_task_wait_iot_comp_cpu_cycle[iot_index]
        observation_all[iot_index, n_task_type * 5 + 1] = task_on_process_local[iot_index]['remain']
        observation_all[iot_index, n_task_type * 5 + 2] = all_task_wait_iot_tran_size[iot_index]
        observation_all[iot_index, n_task_type * 5 + 3] = task_on_transmit_local[iot_index]['remain']
        observation_all[iot_index, n_task_type * 5 + 4] = pull_uplink_rate_iot
        observation_all[iot_index, n_task_type * 5 + 5] = comp_cap_iot[iot_index]
        for i in range(n_edge):
            observation_all[iot_index, n_task_type * 5 + 6 + i] = tran_edge_cap_iot[iot_index][i]
        for i in range(n_cloud):
            observation_all[iot_index, n_task_type * 5 + 6 + n_edge + i] = tran_cloud_cap_iot[iot_index][i]

        observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud] = pull_uplink_rate_edge
        observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud + 1] = 0  
        observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud + 2] = \
        all_task_wait_edge_comp_size_cpu_cycle[iot_index][0]
        observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud + 3] = task_on_process_edge[iot_index][0][
            'remain']

        for n_task_type_index in range(n_task_type):
            observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud + 4 + n_task_type_index] = \
            all_task_wait_edge_comp_count[iot_index][0][n_task_type_index]
            observation_all[iot_index, n_task_type * 6 + 6 + n_edge + n_cloud + 4 + n_task_type_index] = \
            per_task_wait_edge_comp_count_cpu_cycle[iot_index][0][n_task_type_index]
            observation_all[iot_index, n_task_type * 7 + 6 + n_edge + n_cloud + 4 + n_task_type_index] = keep_alive[
                n_task_type_index]  

        observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 4] = pull_uplink_rate_edge
        observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 5] = 0  
        observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 6] = \
        all_task_wait_edge_comp_size_cpu_cycle[iot_index][1]
        observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 7] = task_on_process_edge[iot_index][1][
            'remain']
        for n_task_type_index in range(n_task_type):
            observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 8 + n_task_type_index] = \
            all_task_wait_edge_comp_count[iot_index][1][n_task_type_index]
            observation_all[iot_index, n_task_type * 9 + 6 + n_edge + n_cloud + 8 + n_task_type_index] = \
                per_task_wait_edge_comp_count_cpu_cycle[iot_index][1][n_task_type_index]
            observation_all[iot_index, n_task_type * 10 + 6 + n_edge + n_cloud + 8 + n_task_type_index] = keep_alive[
                n_task_type_index]  
        observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 8] = pull_uplink_rate_cloud
        observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 9] = comp_cap_cloud[0]
        observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 10] = \
        all_task_wait_cloud_comp_size_cpu_cycle[iot_index][0]
        observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 11] = task_on_process_cloud[iot_index][0][
            'remain']
        for n_task_type_index in range(n_task_type):
            observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 12 + n_task_type_index] = \
            all_task_wait_cloud_comp_count[iot_index][0][n_task_type_index]
            observation_all[iot_index, n_task_type * 12 + 6 + n_edge + n_cloud + 12 + n_task_type_index] = \
            per_task_wait_cloud_comp_count_cpu_cycle[iot_index][0][n_task_type_index]
            observation_all[iot_index, n_task_type * 13 + 6 + n_edge + n_cloud + 12 + n_task_type_index] = keep_alive[
                n_task_type_index]  
    return observation_all


def step(
    j_time, action_time,
    n_iot, n_edge, n_cloud, n_task_type, n_features, keep_alive,

    pull_uplink_rate_iot, comp_cap_iot,
    tran_edge_cap_iot, tran_cloud_cap_iot,
    pull_uplink_rate_edge, pull_uplink_rate_cloud,
    comp_cap_cloud, comp_cap_edge,

    task, energy_list, expense_list, unfinish_indi, cold_count_list,
    cold_start_delay, transing_process_delay, computing_process_delay,

    Queue_iot_comp, all_task_wait_iot_comp_count,
    per_task_wait_iot_comp_count_cpu_cycle, all_task_wait_iot_comp_cpu_cycle,

    Queue_iot_tran, all_task_wait_iot_tran_count,
    all_task_wait_iot_tran_edge_count, per_task_wait_iot_tran_count_size,
    all_task_wait_iot_tran_size,

    task_on_transmit_local, task_on_process_local,

    task_on_process_edge, wait_comp_process_delay, Queue_edge_comp,
    delay_list, wait_trans_process_delay,
    esc, all_task_wait_edge_comp_count,
    per_task_wait_edge_comp_count_cpu_cycle,
    all_task_wait_edge_comp_size_cpu_cycle,

    Queue_cloud_comp, all_task_wait_cloud_comp_count,
    per_task_wait_cloud_comp_count_cpu_cycle,
    all_task_wait_cloud_comp_size_cpu_cycle,

    tr_power, task_on_process_cloud):
    j = j_time

    for iot_index in range(n_iot):
        if task[iot_index][j][0]['function_id'] != -1 and action_time[iot_index][j][0] != -2:
            tmp_dict = {'function_id': task[iot_index][j][0]['function_id'],
                        'data_size': task[iot_index][j][0]['data_size'],
                        'cpu_cycle': task[iot_index][j][0]['cpu_cycle'], 'task_arrive_time': j,
                        'max_delay': task[iot_index][j][0]['max_delay']}
            if action_time[iot_index][j][0] == 0:
                Queue_iot_comp[iot_index].put(tmp_dict)
                id_index = int(tmp_dict['function_id'])
                all_task_wait_iot_comp_count[iot_index][id_index] += 1
                per_task_wait_iot_comp_count_cpu_cycle[iot_index][id_index] += tmp_dict['cpu_cycle']
                all_task_wait_iot_comp_cpu_cycle[iot_index] += tmp_dict['cpu_cycle']
            else:
                tmp_tran = {'function_id': task[iot_index][j][0]['function_id'],
                            'data_size': task[iot_index][j][0]['data_size'],
                            'cpu_cycle': task[iot_index][j][0]['cpu_cycle'], 'task_arrive_time': j,
                            'iot_action': action_time[iot_index][j][0], 'max_delay': task[iot_index][j][0]['max_delay']
                            }
                Queue_iot_tran[iot_index].put(tmp_tran)
                id_index = int(tmp_tran['function_id'])
                all_task_wait_iot_tran_count[iot_index][id_index] += 1
                if tmp_tran['iot_action'] == 1:
                    all_task_wait_iot_tran_edge_count[iot_index][0] += 1
                if tmp_tran['iot_action'] == 2:
                    all_task_wait_iot_tran_edge_count[iot_index][1] += 1
                per_task_wait_iot_tran_count_size[iot_index][id_index] += tmp_tran['data_size']
                all_task_wait_iot_tran_size[iot_index] += tmp_tran['data_size']

    edge_count = np.zeros([1, n_edge])
    for edge_index in range(n_edge):
        for iot_index in range(n_iot):
            if (task_on_transmit_local[iot_index]['remain'] > 0 and task_on_transmit_local[iot_index][
                'iot_action'] - 1 == edge_index) or (all_task_wait_iot_tran_edge_count[iot_index][edge_index] != 0 and (
            not Queue_iot_tran[iot_index].empty())) or task_on_process_edge[iot_index][edge_index]['remain'] > 0 or (
            not Queue_edge_comp[iot_index][edge_index].empty()):
                edge_count[0][edge_index] += 1

    for iot_index in range(n_iot):
        slot_time_remain = 1
        while slot_time_remain > 0:
            if task_on_process_local[iot_index]['remain'] == 0 and (not Queue_iot_comp[iot_index].empty()):
                while not Queue_iot_comp[iot_index].empty():

                    get_task = Queue_iot_comp[iot_index].get()
                    id_index = int(get_task['function_id'])
                    time_index = int(get_task['task_arrive_time'])

                    if time_index == 0:
                        print("本地取任务", slot_time_remain, get_task)
                    size_index = get_task['data_size']
                    cpu_index = get_task['cpu_cycle']
                    all_task_wait_iot_comp_count[iot_index][id_index] -= 1
                    per_task_wait_iot_comp_count_cpu_cycle[iot_index][id_index] -= get_task['cpu_cycle']
                    all_task_wait_iot_comp_cpu_cycle[iot_index] -= get_task['cpu_cycle']
                    wait_comp_process_delay[iot_index][time_index][0] = j + 1 - slot_time_remain - get_task[
                        'task_arrive_time']
                    if j + 1 - slot_time_remain - get_task['task_arrive_time'] >= get_task['max_delay']:
                        cold_count_list[iot_index][time_index][0] = 2
                        cold_start_delay[iot_index][time_index][0] = 0
                        computing_process_delay[iot_index][time_index][0] = 0
                        unfinish_indi[iot_index][time_index][0] = 3
                        delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - get_task['task_arrive_time']
                        energy_list[iot_index][time_index][0] = 0
                        expense_list[iot_index][time_index][0] = 0
                    else:
                        task_on_process_local[iot_index]['function_id'] = int(get_task['function_id'])
                        task_on_process_local[iot_index]['data_size'] = get_task['data_size']
                        task_on_process_local[iot_index]['cpu_cycle'] = get_task['cpu_cycle']
                        task_on_process_local[iot_index]['task_arrive_time'] = get_task['task_arrive_time']
                        task_on_process_local[iot_index]['remain'] = get_task['cpu_cycle']
                        task_on_process_local[iot_index]['max_delay'] = get_task['max_delay']

                        if j + 1 - slot_time_remain - iot_pool[iot_index][id_index] < keep_alive[id_index] and \
                                iot_pool[iot_index][id_index] != 0:
                            cold_count_list[iot_index][time_index][0] = 0
                            cold_start_delay[iot_index][time_index][0] = 0
                            task_on_process_local[iot_index]['flag'] = 0
                            task_on_process_local[iot_index]['block'] = 0
                            iot_pool[iot_index][id_index] = j + 1 - slot_time_remain
                        if j + 1 - slot_time_remain - iot_pool[iot_index][id_index] >= keep_alive[id_index] or \
                                iot_pool[iot_index][id_index] == 0:
                            cold_count_list[iot_index][time_index][0] = 1
                            cold_start_delay[iot_index][time_index][0] = task_on_process_local[iot_index][
                                                                             'data_size'] / pull_uplink_rate_iot
                            task_on_process_local[iot_index]['flag'] = cold_start_delay[iot_index][time_index][0]
                        break

            if slot_time_remain > 0 and task_on_process_local[iot_index]['remain'] > 0 and \
                    task_on_process_local[iot_index]['flag'] > 0:
                task_on_process_local[iot_index]['flag'] -= slot_time_remain
                time_index = int(task_on_process_local[iot_index]['task_arrive_time'])
                if time_index == 0:
                    print("本地冷启动", slot_time_remain)
                id_index = int(task_on_process_local[iot_index]['function_id'])
                if task_on_process_local[iot_index]['flag'] > 0:  
                    slot_time_remain = 0
                if task_on_process_local[iot_index]['flag'] <= 0:
                    slot_time_remain = -task_on_process_local[iot_index]['flag']
                    if j + 1 - slot_time_remain - task_on_process_local[iot_index]['task_arrive_time'] >= \
                            task_on_process_local[iot_index]['max_delay']:
                        unfinish_indi[iot_index][time_index][0] = 4
                        computing_process_delay[iot_index][time_index][0] = 0
                        delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                               task_on_process_local[iot_index]['task_arrive_time']
                        energy_list[iot_index][time_index][0] = esc * (comp_cap_iot[iot_index] ** 2) * ((
                                    computing_process_delay[iot_index][time_index][0] +
                                    cold_start_delay[iot_index][time_index][0]))
                        expense_list[iot_index][time_index][0] = 0
                        task_on_process_local[iot_index]['function_id'] = -1
                        task_on_process_local[iot_index]['data_size'] = 0
                        task_on_process_local[iot_index]['cpu_cycle'] = 0
                        task_on_process_local[iot_index]['task_arrive_time'] = 0
                        task_on_process_local[iot_index]['remain'] = 0
                        task_on_process_local[iot_index]['flag'] = 0
                        task_on_process_local[iot_index]['max_delay'] = 0
                        task_on_process_local[iot_index]['block'] = 0
                        iot_pool[iot_index][id_index] = j + 1 - slot_time_remain  
                    else:
                        task_on_process_local[iot_index]['flag'] = 0
                        iot_pool[iot_index][id_index] = j + 1 - slot_time_remain
            if slot_time_remain > 0 and task_on_process_local[iot_index]['remain'] > 0 and \
                    task_on_process_local[iot_index]['flag'] == 0:
                id_index = int(task_on_process_local[iot_index]['function_id'])
                time_index = int(task_on_process_local[iot_index]['task_arrive_time'])
                if time_index == 0:
                    print("本地执行", slot_time_remain)
                task_remain_time = task_on_process_local[iot_index]['max_delay'] - (
                            j + 1 - slot_time_remain - task_on_process_local[iot_index]['task_arrive_time'])
                if task_remain_time >= slot_time_remain: 
                    task_on_process_local[iot_index]['remain'] = task_on_process_local[iot_index]['remain'] - \
                                                                 comp_cap_iot[iot_index] * slot_time_remain
                    if task_on_process_local[iot_index]['remain'] > 0:
                        slot_time_remain = 0
                        iot_pool[iot_index][id_index] = j + 1 - slot_time_remain
                    if task_on_process_local[iot_index]['remain'] <= 0:
                        task_on_process_local[iot_index]['remain'] = task_on_process_local[iot_index]['remain'] + \
                                                                     comp_cap_iot[iot_index] * slot_time_remain
                        task_remain_need_time = task_on_process_local[iot_index]['remain'] / comp_cap_iot[iot_index]
                        task_on_process_local[iot_index]['remain'] = 0
                        slot_time_remain = slot_time_remain - task_remain_need_time
                        unfinish_indi[iot_index][time_index][0] = 0
                        computing_process_delay[iot_index][time_index][0] = task_on_process_local[iot_index][
                                                                                'cpu_cycle'] / comp_cap_iot[iot_index]
                        delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                               task_on_process_local[iot_index]['task_arrive_time']
                        energy_list[iot_index][time_index][0] = esc * (comp_cap_iot[iot_index] ** 2) * ((
                                    computing_process_delay[iot_index][time_index][0] +
                                    cold_start_delay[iot_index][time_index][0]))
                        expense_list[iot_index][time_index][0] = 0
                        task_on_process_local[iot_index]['function_id'] = -1
                        task_on_process_local[iot_index]['data_size'] = 0
                        task_on_process_local[iot_index]['cpu_cycle'] = 0
                        task_on_process_local[iot_index]['task_arrive_time'] = 0
                        task_on_process_local[iot_index]['remain'] = 0
                        task_on_process_local[iot_index]['flag'] = 0
                        task_on_process_local[iot_index]['max_delay'] = 0
                        task_on_process_local[iot_index]['block'] = 0
                        iot_pool[iot_index][id_index] = j + 1 - slot_time_remain
                if task_remain_time < slot_time_remain:
                    task_on_process_local[iot_index]['remain'] = task_on_process_local[iot_index]['remain'] - \
                                                                 comp_cap_iot[iot_index] * task_remain_time

                    if task_on_process_local[iot_index]['remain'] <= 0:
                        if task_on_process_local[iot_index]['remain'] == 0:
                            slot_time_remain = slot_time_remain - task_remain_time
                        if task_on_process_local[iot_index]['remain'] < 0:
                            task_on_process_local[iot_index]['remain'] = task_on_process_local[iot_index]['remain'] + \
                                                                         comp_cap_iot[iot_index] * task_remain_time
                            task_remain_need_time = task_on_process_local[iot_index]['remain'] / comp_cap_iot[iot_index]
                            task_on_process_local[iot_index]['remain'] = 0
                            slot_time_remain = slot_time_remain - task_remain_need_time
                        unfinish_indi[iot_index][time_index][0] = 0
                        computing_process_delay[iot_index][time_index][0] = task_on_process_local[iot_index][
                                                                                'cpu_cycle'] / comp_cap_iot[iot_index]
                        delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                               task_on_process_local[iot_index]['task_arrive_time']
                        energy_list[iot_index][time_index][0] = esc * (comp_cap_iot[iot_index] ** 2) * ((
                                    computing_process_delay[iot_index][time_index][0] +
                                    cold_start_delay[iot_index][time_index][0]))
                        expense_list[iot_index][time_index][0] = 0
                        task_on_process_local[iot_index]['function_id'] = -1
                        task_on_process_local[iot_index]['data_size'] = 0
                        task_on_process_local[iot_index]['cpu_cycle'] = 0
                        task_on_process_local[iot_index]['task_arrive_time'] = 0
                        task_on_process_local[iot_index]['remain'] = 0
                        task_on_process_local[iot_index]['flag'] = 0
                        task_on_process_local[iot_index]['max_delay'] = 0
                        task_on_process_local[iot_index]['block'] = 0
                        iot_pool[iot_index][id_index] = j + 1 - slot_time_remain
                    if task_on_process_local[iot_index]['remain'] > 0:
                        slot_time_remain = slot_time_remain - task_remain_time
                        unfinish_indi[iot_index][time_index][0] = 5
                        computing_process_delay[iot_index][time_index][0] = (task_on_process_local[iot_index][
                                                                                 'cpu_cycle'] -
                                                                             task_on_process_local[iot_index][
                                                                                 'remain']) / comp_cap_iot[iot_index]
                        delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                               task_on_process_local[iot_index]['task_arrive_time']
                        energy_list[iot_index][time_index][0] = esc * (comp_cap_iot[iot_index] ** 2) * ((
                                    computing_process_delay[iot_index][time_index][0] +
                                    cold_start_delay[iot_index][time_index][0]))
                        expense_list[iot_index][time_index][0] = 0
                        task_on_process_local[iot_index]['function_id'] = -1
                        task_on_process_local[iot_index]['data_size'] = 0
                        task_on_process_local[iot_index]['cpu_cycle'] = 0
                        task_on_process_local[iot_index]['task_arrive_time'] = 0
                        task_on_process_local[iot_index]['remain'] = 0
                        task_on_process_local[iot_index]['flag'] = 0
                        task_on_process_local[iot_index]['max_delay'] = 0
                        task_on_process_local[iot_index]['block'] = 0
                        iot_pool[iot_index][id_index] = j + 1 - slot_time_remain
            if task_on_process_local[iot_index]['remain'] == 0 and Queue_iot_comp[iot_index].empty() and \
                    task_on_process_local[iot_index]['flag'] == 0:
                slot_time_remain = 0

    for iot_index in range(n_iot):
        slot_time_remain = 1
        while slot_time_remain > 0:
            if task_on_transmit_local[iot_index]['function_id'] == -1 and task_on_transmit_local[iot_index][
                'remain'] == 0 and (not Queue_iot_tran[iot_index].empty()):
                while not Queue_iot_tran[iot_index].empty():
                    get_task = Queue_iot_tran[iot_index].get()
                    id_index = int(get_task['function_id'])
                    time_index = int(get_task['task_arrive_time'])
                    all_task_wait_iot_tran_count[iot_index][id_index] -= 1
                    if get_task['iot_action'] == 1:
                        all_task_wait_iot_tran_edge_count[iot_index][0] -= 1
                    if get_task['iot_action'] == 2:
                        all_task_wait_iot_tran_edge_count[iot_index][1] -= 1
                    per_task_wait_iot_tran_count_size[iot_index][id_index] -= get_task['data_size']
                    all_task_wait_iot_tran_size[iot_index] -= get_task['data_size']
                    wait_trans_process_delay[iot_index][time_index][0] = j + 1 - slot_time_remain - get_task[
                        'task_arrive_time']
                    if wait_trans_process_delay[iot_index][time_index][0] >= get_task['max_delay']:
                        transing_process_delay[iot_index][time_index][0] = 0
                        cold_count_list[iot_index][time_index][0] = 2
                        cold_start_delay[iot_index][time_index][0] = 0
                        computing_process_delay[iot_index][time_index][0] = 0
                        unfinish_indi[iot_index][time_index][0] = 1
                        delay_list[iot_index][time_index][0] = wait_trans_process_delay[iot_index][time_index][0]
                        energy_list[iot_index][time_index][0] = 0
                        expense_list[iot_index][time_index][0] = 0
                    else:
                        task_on_transmit_local[iot_index]['function_id'] = int(get_task['function_id'])
                        task_on_transmit_local[iot_index]['data_size'] = get_task['data_size']
                        task_on_transmit_local[iot_index]['cpu_cycle'] = get_task['cpu_cycle']
                        task_on_transmit_local[iot_index]['max_delay'] = get_task['max_delay']
                        task_on_transmit_local[iot_index]['task_arrive_time'] = get_task['task_arrive_time']
                        task_on_transmit_local[iot_index]['iot_action'] = get_task['iot_action']
                        task_on_transmit_local[iot_index]['remain'] = get_task['data_size']
                        break

            if slot_time_remain > 0 and task_on_transmit_local[iot_index]['remain'] > 0 and \
                    task_on_transmit_local[iot_index]['function_id'] != -1:
                tran_index_flag = int(task_on_transmit_local[iot_index]['iot_action'])
                if tran_index_flag == 1 or tran_index_flag == 2:
                    tran_index = tran_index_flag - 1
                    tran_iot_camp = tran_edge_cap_iot[iot_index][tran_index]
                elif tran_index_flag == 3:
                    tran_index = tran_index_flag - 3
                    tran_iot_camp = tran_cloud_cap_iot[iot_index][tran_index]
                else:
                    print("传输的动作不对")
                id_index = int(task_on_transmit_local[iot_index]['function_id'])
                time_index = int(task_on_transmit_local[iot_index]['task_arrive_time'])
                if time_index == 0:
                    print("本地取传输执行", slot_time_remain, task_on_transmit_local[iot_index])
                task_remain_time = task_on_transmit_local[iot_index]['max_delay'] - (
                            j + 1 - slot_time_remain - task_on_transmit_local[iot_index]['task_arrive_time'])
                if task_remain_time > slot_time_remain and task_on_transmit_local[iot_index]['function_id'] != -1:
                    task_on_transmit_local[iot_index]['remain'] = task_on_transmit_local[iot_index][
                                                                      'remain'] - tran_iot_camp * slot_time_remain
                    if task_on_transmit_local[iot_index]['remain'] > 0:
                        slot_time_remain = 0
                    if task_on_transmit_local[iot_index]['remain'] <= 0:  
                        task_on_transmit_local[iot_index]['remain'] = task_on_transmit_local[iot_index][
                                                                          'remain'] + tran_iot_camp * slot_time_remain
                        task_remain_need_time = task_on_transmit_local[iot_index]['remain'] / tran_iot_camp
                        task_on_transmit_local[iot_index]['remain'] = 0
                        slot_time_remain = slot_time_remain - task_remain_need_time
                        tmp_tran = {'function_id': task_on_transmit_local[iot_index]['function_id'],
                                    'data_size': task_on_transmit_local[iot_index]['data_size'],
                                    'cpu_cycle': task_on_transmit_local[iot_index]['cpu_cycle'],
                                    'task_arrive_time': task_on_transmit_local[iot_index]['task_arrive_time'],
                                    'max_delay': task_on_transmit_local[iot_index]['max_delay'],
                                    'tran_finish_time': (j + 1 - slot_time_remain)
                                    }

                        transing_process_delay[iot_index][time_index][0] = (j + 1 - slot_time_remain -
                                                                            task_on_transmit_local[iot_index][
                                                                                'task_arrive_time']
                                                                            - wait_trans_process_delay[iot_index][
                                                                                time_index][0])
                        id_index = tmp_tran['function_id']
                        cpu_size = tmp_tran['cpu_cycle']
                        if tran_index_flag == 1 or tran_index_flag == 2:
                            tran_index = tran_index_flag - 1
                            Queue_edge_comp[iot_index][tran_index].put(tmp_tran)
                            all_task_wait_edge_comp_count[iot_index][tran_index][id_index] += 1
                            per_task_wait_edge_comp_count_cpu_cycle[iot_index][tran_index][id_index] += cpu_size
                            all_task_wait_edge_comp_size_cpu_cycle[iot_index][tran_index] += cpu_size
                        elif tran_index_flag == 3:
                            tran_index = tran_index_flag - 3
                            Queue_cloud_comp[iot_index][tran_index].put(tmp_tran)
                            all_task_wait_cloud_comp_count[iot_index][tran_index][id_index] += 1
                            per_task_wait_cloud_comp_count_cpu_cycle[iot_index][tran_index][id_index] += cpu_size
                            all_task_wait_cloud_comp_size_cpu_cycle[iot_index][tran_index] += cpu_size
                        else:
                            print("false")
                        task_on_transmit_local[iot_index]['function_id'] = -1
                        task_on_transmit_local[iot_index]['data_size'] = 0
                        task_on_transmit_local[iot_index]['cpu_cycle'] = 0
                        task_on_transmit_local[iot_index]['task_arrive_time'] = 0
                        task_on_transmit_local[iot_index]['iot_action'] = -1
                        task_on_transmit_local[iot_index]['max_delay'] = 0
                        task_on_transmit_local[iot_index]['remain'] = 0
                if task_remain_time <= slot_time_remain and task_on_transmit_local[iot_index]['function_id'] != -1:
                    task_on_transmit_local[iot_index]['remain'] = task_on_transmit_local[iot_index][
                                                                      'remain'] - tran_iot_camp * task_remain_time
                    if task_on_transmit_local[iot_index]['remain'] < 0:
                        task_on_transmit_local[iot_index]['remain'] = task_on_transmit_local[iot_index][
                                                                          'remain'] + tran_iot_camp * task_remain_time
                        task_remain_need_time = task_on_transmit_local[iot_index]['remain'] / tran_iot_camp
                        task_on_transmit_local[iot_index]['remain'] = 0
                        slot_time_remain = slot_time_remain - task_remain_need_time
                        tmp_tran = {'function_id': task_on_transmit_local[iot_index]['function_id'],
                                    'data_size': task_on_transmit_local[iot_index]['data_size'],
                                    'cpu_cycle': task_on_transmit_local[iot_index]['cpu_cycle'],
                                    'task_arrive_time': task_on_transmit_local[iot_index]['task_arrive_time'],
                                    'max_delay': task_on_transmit_local[iot_index]['max_delay'],
                                    'tran_finish_time': (j + 1 - slot_time_remain)
                                    }
                        transing_process_delay[iot_index][time_index][0] = (j + 1 - slot_time_remain -
                                                                            task_on_transmit_local[iot_index][
                                                                                'task_arrive_time']
                                                                            - wait_trans_process_delay[iot_index][
                                                                                time_index][0])
                        id_index = tmp_tran['function_id']
                        cpu_size = tmp_tran['cpu_cycle']
                        if tran_index_flag == 1 or tran_index_flag == 2:
                            tran_index = tran_index_flag - 1
                            Queue_edge_comp[iot_index][tran_index].put(tmp_tran)
                            all_task_wait_edge_comp_count[iot_index][tran_index][id_index] += 1
                            per_task_wait_edge_comp_count_cpu_cycle[iot_index][tran_index][id_index] += cpu_size
                            all_task_wait_edge_comp_size_cpu_cycle[iot_index][tran_index] += cpu_size
                        elif tran_index_flag == 3:
                            tran_index = tran_index_flag - 3
                            Queue_cloud_comp[iot_index][tran_index].put(tmp_tran)
                            all_task_wait_cloud_comp_count[iot_index][tran_index][id_index] += 1
                            per_task_wait_cloud_comp_count_cpu_cycle[iot_index][tran_index][id_index] += cpu_size
                            all_task_wait_cloud_comp_size_cpu_cycle[iot_index][tran_index] += cpu_size
                        else:
                            print("false")
                        task_on_transmit_local[iot_index]['function_id'] = -1
                        task_on_transmit_local[iot_index]['data_size'] = 0
                        task_on_transmit_local[iot_index]['cpu_cycle'] = 0
                        task_on_transmit_local[iot_index]['task_arrive_time'] = 0
                        task_on_transmit_local[iot_index]['iot_action'] = -1
                        task_on_transmit_local[iot_index]['remain'] = 0
                        task_on_transmit_local[iot_index]['max_delay'] = 0

                    if task_on_transmit_local[iot_index]['remain'] >= 0:
                        slot_time_remain = slot_time_remain - task_remain_time
                        transing_process_delay[iot_index][time_index][0] = (j + 1 - slot_time_remain -
                                                                            task_on_transmit_local[iot_index][
                                                                                'task_arrive_time']
                                                                            - wait_trans_process_delay[iot_index][
                                                                                time_index][0])
                        unfinish_indi[iot_index][time_index][0] = 2  
                        computing_process_delay[iot_index][time_index][0] = 0
                        delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                               task_on_transmit_local[iot_index]['task_arrive_time']
                        energy_list[iot_index][time_index][0] = tr_power * \
                                                                transing_process_delay[iot_index][time_index][0]
                        expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                 computing_process_delay[iot_index][time_index][
                                                                     0] + 6e-5
                        task_on_transmit_local[iot_index]['function_id'] = -1
                        task_on_transmit_local[iot_index]['data_size'] = 0
                        task_on_transmit_local[iot_index]['cpu_cycle'] = 0
                        task_on_transmit_local[iot_index]['task_arrive_time'] = 0
                        task_on_transmit_local[iot_index]['iot_action'] = -1
                        task_on_transmit_local[iot_index]['remain'] = 0
                        task_on_transmit_local[iot_index]['max_delay'] = 0
            if task_on_transmit_local[iot_index]['function_id'] == -1 and task_on_transmit_local[iot_index][
                'remain'] == 0 and Queue_iot_tran[iot_index].empty():
                slot_time_remain = 0

    for iot_index in range(n_iot):
        for edge_index in range(n_edge):
            slot_time_remain = 1
            while slot_time_remain > 0:
                if task_on_process_edge[iot_index][edge_index]['remain'] == 0 and Queue_edge_comp[iot_index][
                    edge_index].empty():
                    slot_time_remain = 0
                if task_on_process_edge[iot_index][edge_index]['remain'] == 0 and (
                not Queue_edge_comp[iot_index][edge_index].empty()):
                    while not Queue_edge_comp[iot_index][edge_index].empty():
                        get_task = Queue_edge_comp[iot_index][edge_index].get()
                        id_index = int(get_task['function_id'])
                        time_index = int(get_task['task_arrive_time'])
                        if time_index == 0:
                            print(f"边缘取任务{edge_index}", slot_time_remain, get_task)
                        if j + 1 - slot_time_remain <= get_task['tran_finish_time']:
                            slot_time_remain = j + 1 - get_task['tran_finish_time']
                        else:
                            slot_time_remain = slot_time_remain
                        if j + 1 - slot_time_remain < get_task['tran_finish_time']:
                            return
                        all_task_wait_edge_comp_count[iot_index][edge_index][id_index] -= 1
                        per_task_wait_edge_comp_count_cpu_cycle[iot_index][edge_index][id_index] -= get_task[
                            'cpu_cycle']
                        all_task_wait_edge_comp_size_cpu_cycle[iot_index][edge_index] -= get_task['cpu_cycle']
                        wait_comp_process_delay[iot_index][time_index][0] = j + 1 - slot_time_remain - get_task[
                            'tran_finish_time']
                        if j + 1 - slot_time_remain - get_task['task_arrive_time'] >= get_task['max_delay']:
                            cold_count_list[iot_index][time_index][0] = 2
                            cold_start_delay[iot_index][time_index][0] = 0
                            computing_process_delay[iot_index][time_index][0] = 0
                            unfinish_indi[iot_index][time_index][0] = 3
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - get_task[
                                'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * ((comp_cap_edge[edge_index] / edge_count[0][edge_index]) ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                        else:
                            task_on_process_edge[iot_index][edge_index]['function_id'] = get_task['function_id']
                            task_on_process_edge[iot_index][edge_index]['data_size'] = get_task['function_id']
                            task_on_process_edge[iot_index][edge_index]['cpu_cycle'] = get_task['cpu_cycle']
                            task_on_process_edge[iot_index][edge_index]['max_delay'] = get_task['max_delay']
                            task_on_process_edge[iot_index][edge_index]['task_arrive_time'] = get_task[
                                'task_arrive_time']
                            task_on_process_edge[iot_index][edge_index]['tran_finish_time'] = get_task[
                                'tran_finish_time']
                            task_on_process_edge[iot_index][edge_index]['remain'] = get_task['cpu_cycle']
                            if j + 1 - slot_time_remain - edge_pool[edge_index][iot_index][id_index] < keep_alive[
                                id_index] and edge_pool[edge_index][iot_index][id_index] != 0:
                                cold_count_list[iot_index][time_index][0] = 0
                                cold_start_delay[iot_index][time_index][0] = 0
                                task_on_process_edge[iot_index][edge_index]['flag'] = 0
                                edge_pool[edge_index][iot_index][id_index] = np.max(
                                    [j + 1 - slot_time_remain, edge_pool[edge_index][iot_index][id_index]])
                            if j + 1 - slot_time_remain - edge_pool[edge_index][iot_index][id_index] >= keep_alive[
                                id_index] or edge_pool[edge_index][iot_index][id_index] == 0:
                                cold_count_list[iot_index][time_index][0] = 1
                                cold_start_delay[iot_index][time_index][0] = \
                                task_on_process_edge[iot_index][edge_index]['data_size'] / pull_uplink_rate_edge
                                task_on_process_edge[iot_index][edge_index]['flag'] = \
                                cold_start_delay[iot_index][time_index][0]
                            break
                if slot_time_remain > 0 and task_on_process_edge[iot_index][edge_index]['remain'] > 0 and \
                        task_on_process_edge[iot_index][edge_index]['flag'] > 0:
                    task_on_process_edge[iot_index][edge_index]['flag'] -= slot_time_remain
                    id_index = task_on_process_edge[iot_index][edge_index]['function_id']
                    time_index = task_on_process_edge[iot_index][edge_index]['task_arrive_time']
                    if time_index == 0:
                        print("边缘取冷启动", slot_time_remain)
                    if task_on_process_edge[iot_index][edge_index]['flag'] > 0:
                        slot_time_remain = 0
                    if task_on_process_edge[iot_index][edge_index]['flag'] <= 0:
                        slot_time_remain = -task_on_process_edge[iot_index][edge_index]['flag']
                        if j + 1 - slot_time_remain - task_on_process_edge[iot_index][edge_index]['task_arrive_time'] >= \
                                task_on_process_edge[iot_index][edge_index]['max_delay']:
                            unfinish_indi[iot_index][time_index][0] = 4
                            computing_process_delay[iot_index][time_index][0] = 0
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                                   task_on_process_edge[iot_index][edge_index][
                                                                       'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * ((comp_cap_edge[edge_index] / edge_count[0][edge_index]) ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                            task_on_process_edge[iot_index][edge_index]['function_id'] = -1
                            task_on_process_edge[iot_index][edge_index]['data_size'] = 0
                            task_on_process_edge[iot_index][edge_index]['cpu_cycle'] = 0
                            task_on_process_edge[iot_index][edge_index]['task_arrive_time'] = 0
                            task_on_process_edge[iot_index][edge_index]['tran_finish_time'] = 0
                            task_on_process_edge[iot_index][edge_index]['flag'] = 0
                            task_on_process_edge[iot_index][edge_index]['remain'] = 0
                            task_on_process_edge[iot_index][edge_index]['max_delay'] = 0
                            edge_pool[edge_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, edge_pool[edge_index][iot_index][id_index]])
                        else:
                            task_on_process_edge[iot_index][edge_index]['flag'] = 0
                            edge_pool[edge_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, edge_pool[edge_index][iot_index][id_index]])
                if slot_time_remain > 0 and task_on_process_edge[iot_index][edge_index]['remain'] > 0 and \
                        task_on_process_edge[iot_index][edge_index]['flag'] == 0:
                    id_index = task_on_process_edge[iot_index][edge_index]['function_id']
                    time_index = task_on_process_edge[iot_index][edge_index]['task_arrive_time']
                    if time_index == 0:
                        print("边缘执行", slot_time_remain)
                    task_remain_time = task_on_process_edge[iot_index][edge_index]['max_delay'] - (
                                j + 1 - slot_time_remain - task_on_process_edge[iot_index][edge_index][
                            'task_arrive_time'])
                    if edge_count[0][edge_index] == 0:
                        print("错误的，3", slot_time_remain,
                              task_on_process_edge[iot_index][edge_index]['remain'],
                              (not Queue_edge_comp[iot_index][edge_index].empty()))
                    if task_remain_time >= slot_time_remain:
                        task_on_process_edge[iot_index][edge_index]['remain'] = \
                        task_on_process_edge[iot_index][edge_index]['remain'] - (
                                    comp_cap_edge[edge_index] / edge_count[0][edge_index]) * slot_time_remain
                        if task_on_process_edge[iot_index][edge_index]['remain'] > 0:
                            slot_time_remain = 0
                            edge_pool[edge_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, edge_pool[edge_index][iot_index][id_index]])
                        if task_on_process_edge[iot_index][edge_index]['remain'] <= 0:
                            task_on_process_edge[iot_index][edge_index]['remain'] = \
                            task_on_process_edge[iot_index][edge_index]['remain'] + (
                                        comp_cap_edge[edge_index] / edge_count[0][edge_index]) * slot_time_remain
                            task_remain_need_time = task_on_process_edge[iot_index][edge_index]['remain'] / (
                                        comp_cap_edge[edge_index] / edge_count[0][edge_index])
                            task_on_process_edge[iot_index][edge_index]['remain'] = 0
                            slot_time_remain = slot_time_remain - task_remain_need_time
                            unfinish_indi[iot_index][time_index][0] = 0
                            computing_process_delay[iot_index][time_index][0] = \
                            task_on_process_edge[iot_index][edge_index]['cpu_cycle'] / (
                                        comp_cap_edge[edge_index] / edge_count[0][edge_index])
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                                   task_on_process_edge[iot_index][edge_index][
                                                                       'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * ((comp_cap_edge[edge_index] / edge_count[0][edge_index]) ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                            task_on_process_edge[iot_index][edge_index]['function_id'] = -1
                            task_on_process_edge[iot_index][edge_index]['data_size'] = 0
                            task_on_process_edge[iot_index][edge_index]['cpu_cycle'] = 0
                            task_on_process_edge[iot_index][edge_index]['task_arrive_time'] = 0
                            task_on_process_edge[iot_index][edge_index]['tran_finish_time'] = 0
                            task_on_process_edge[iot_index][edge_index]['flag'] = 0
                            task_on_process_edge[iot_index][edge_index]['remain'] = 0
                            task_on_process_edge[iot_index][edge_index]['max_delay'] = 0
                            edge_pool[edge_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, edge_pool[edge_index][iot_index][id_index]])
                    if task_remain_time < slot_time_remain:
                        task_on_process_edge[iot_index][edge_index]['remain'] = \
                        task_on_process_edge[iot_index][edge_index]['remain'] - (
                                    comp_cap_edge[edge_index] / edge_count[0][edge_index]) * task_remain_time

                        if task_on_process_edge[iot_index][edge_index]['remain'] <= 0:
                            if task_on_process_edge[iot_index][edge_index]['remain'] == 0:
                                slot_time_remain = slot_time_remain - task_remain_time
                            if task_on_process_edge[iot_index][edge_index]['remain'] < 0:
                                task_on_process_edge[iot_index][edge_index]['remain'] = \
                                task_on_process_edge[iot_index][edge_index]['remain'] + (
                                            comp_cap_edge[edge_index] / edge_count[0][edge_index]) * task_remain_time
                                task_remain_need_time = task_on_process_edge[iot_index][edge_index]['remain'] / (
                                            comp_cap_edge[edge_index] / edge_count[0][edge_index])
                                if edge_count[0][edge_index] == 0:
                                    print("错误的，5")
                                task_on_process_edge[iot_index][edge_index]['remain'] = 0
                                slot_time_remain = slot_time_remain - task_remain_need_time
                            unfinish_indi[iot_index][time_index][0] = 0
                            computing_process_delay[iot_index][time_index][0] = \
                            task_on_process_edge[iot_index][edge_index]['cpu_cycle'] / (
                                        comp_cap_edge[edge_index] / edge_count[0][edge_index])
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                                   task_on_process_edge[iot_index][edge_index][
                                                                       'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * ((comp_cap_edge[edge_index] / edge_count[0][edge_index]) ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            if edge_count[0][edge_index] == 0:
                                print("错误的，6")
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                            task_on_process_edge[iot_index][edge_index]['function_id'] = -1
                            task_on_process_edge[iot_index][edge_index]['data_size'] = 0
                            task_on_process_edge[iot_index][edge_index]['cpu_cycle'] = 0
                            task_on_process_edge[iot_index][edge_index]['task_arrive_time'] = 0
                            task_on_process_edge[iot_index][edge_index]['tran_finish_time'] = 0
                            task_on_process_edge[iot_index][edge_index]['flag'] = 0
                            task_on_process_edge[iot_index][edge_index]['remain'] = 0
                            task_on_process_edge[iot_index][edge_index]['max_delay'] = 0
                            edge_pool[edge_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, edge_pool[edge_index][iot_index][id_index]])
                        if task_on_process_edge[iot_index][edge_index]['remain'] > 0:

                            slot_time_remain = slot_time_remain - task_remain_time
                            unfinish_indi[iot_index][time_index][0] = 5
                            computing_process_delay[iot_index][time_index][0] = (task_on_process_edge[iot_index][
                                                                                     edge_index]['cpu_cycle'] -
                                                                                 task_on_process_edge[iot_index][
                                                                                     edge_index]['remain']) / (
                                                                                            comp_cap_edge[edge_index] /
                                                                                            edge_count[0][edge_index])
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                                   task_on_process_edge[iot_index][edge_index][
                                                                       'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * ((comp_cap_edge[edge_index] / edge_count[0][edge_index]) ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            if edge_count[0][edge_index] == 0:
                                print("错误的，7")
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                            task_on_process_edge[iot_index][edge_index]['function_id'] = -1
                            task_on_process_edge[iot_index][edge_index]['data_size'] = 0
                            task_on_process_edge[iot_index][edge_index]['cpu_cycle'] = 0
                            task_on_process_edge[iot_index][edge_index]['task_arrive_time'] = 0
                            task_on_process_edge[iot_index][edge_index]['tran_finish_time'] = 0
                            task_on_process_edge[iot_index][edge_index]['flag'] = 0
                            task_on_process_edge[iot_index][edge_index]['remain'] = 0
                            task_on_process_edge[iot_index][edge_index]['max_delay'] = 0
                            edge_pool[edge_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, edge_pool[edge_index][iot_index][id_index]])

    for iot_index in range(n_iot):
        for cloud_index in range(n_cloud):
            slot_time_remain = 1
            while slot_time_remain > 0:
                if task_on_process_cloud[iot_index][cloud_index]['remain'] == 0 and Queue_cloud_comp[iot_index][
                    cloud_index].empty():
                    slot_time_remain = 0
                if task_on_process_cloud[iot_index][cloud_index]['remain'] == 0 and (
                not Queue_cloud_comp[iot_index][cloud_index].empty()):
                    while not Queue_cloud_comp[iot_index][cloud_index].empty():
                        get_task = Queue_cloud_comp[iot_index][cloud_index].get()
                        id_index = int(get_task['function_id'])
                        time_index = int(get_task['task_arrive_time'])
                        if time_index == 0:
                            print("云取任务", slot_time_remain)
                        if j + 1 - slot_time_remain <= get_task['tran_finish_time']:
                            slot_time_remain = j + 1 - get_task['tran_finish_time']
                        else:
                            slot_time_remain = slot_time_remain
                        if j + 1 - slot_time_remain < get_task['tran_finish_time']:
                            return
                        all_task_wait_cloud_comp_count[iot_index][cloud_index][id_index] -= 1
                        per_task_wait_cloud_comp_count_cpu_cycle[iot_index][cloud_index][id_index] -= get_task[
                            'cpu_cycle']
                        all_task_wait_cloud_comp_size_cpu_cycle[iot_index][cloud_index] -= get_task['cpu_cycle']
                        wait_comp_process_delay[iot_index][time_index][0] = j + 1 - slot_time_remain - get_task[
                            'tran_finish_time']
                        if j + 1 - slot_time_remain - get_task['task_arrive_time'] >= get_task['max_delay']:
                            cold_count_list[iot_index][time_index][0] = 2
                            cold_start_delay[iot_index][time_index][0] = 0
                            computing_process_delay[iot_index][time_index][0] = 0
                            unfinish_indi[iot_index][time_index][0] = 3
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - get_task[
                                'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * (comp_cap_cloud[cloud_index] ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                        else:
                            task_on_process_cloud[iot_index][cloud_index]['function_id'] = get_task['function_id']
                            task_on_process_cloud[iot_index][cloud_index]['data_size'] = get_task['function_id']
                            task_on_process_cloud[iot_index][cloud_index]['cpu_cycle'] = get_task['cpu_cycle']
                            task_on_process_cloud[iot_index][cloud_index]['max_delay'] = get_task['max_delay']
                            task_on_process_cloud[iot_index][cloud_index]['task_arrive_time'] = get_task[
                                'task_arrive_time']
                            task_on_process_cloud[iot_index][cloud_index]['tran_finish_time'] = get_task[
                                'tran_finish_time']
                            task_on_process_cloud[iot_index][cloud_index]['remain'] = get_task['cpu_cycle']
                            if j + 1 - slot_time_remain - cloud_pool[cloud_index][iot_index][id_index] < keep_alive[
                                id_index] and cloud_pool[cloud_index][iot_index][id_index] != 0:
                                cold_count_list[iot_index][time_index][0] = 0
                                cold_start_delay[iot_index][time_index][0] = 0
                                task_on_process_cloud[iot_index][cloud_index]['flag'] = 0
                                cloud_pool[cloud_index][iot_index][id_index] = np.max(
                                    [j + 1 - slot_time_remain, cloud_pool[cloud_index][iot_index][id_index]])

                            if j + 1 - slot_time_remain - cloud_pool[cloud_index][iot_index][id_index] >= keep_alive[
                                id_index] or cloud_pool[cloud_index][iot_index][id_index] == 0:
                                cold_count_list[iot_index][time_index][0] = 1
                                cold_start_delay[iot_index][time_index][0] = \
                                task_on_process_cloud[iot_index][cloud_index]['data_size'] / pull_uplink_rate_cloud
                                task_on_process_cloud[iot_index][cloud_index]['flag'] = \
                                cold_start_delay[iot_index][time_index][0]
                            break
                if slot_time_remain > 0 and task_on_process_cloud[iot_index][cloud_index]['remain'] > 0 and \
                        task_on_process_cloud[iot_index][cloud_index]['flag'] > 0:
                    task_on_process_cloud[iot_index][cloud_index]['flag'] -= slot_time_remain
                    id_index = task_on_process_cloud[iot_index][cloud_index]['function_id']
                    time_index = task_on_process_cloud[iot_index][cloud_index]['task_arrive_time']
                    if time_index == 0:
                        print("云取冷启动", slot_time_remain)
                    if task_on_process_cloud[iot_index][cloud_index]['flag'] > 0:
                        slot_time_remain = 0
                    if task_on_process_cloud[iot_index][cloud_index]['flag'] <= 0:
                        slot_time_remain = -task_on_process_cloud[iot_index][cloud_index]['flag']
                        if j + 1 - slot_time_remain - task_on_process_cloud[iot_index][cloud_index][
                            'task_arrive_time'] >= task_on_process_cloud[iot_index][cloud_index]['max_delay']:
                            unfinish_indi[iot_index][time_index][0] = 4
                            computing_process_delay[iot_index][time_index][0] = 0
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                                   task_on_process_cloud[iot_index][cloud_index][
                                                                       'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * (comp_cap_cloud[cloud_index] ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                            task_on_process_cloud[iot_index][cloud_index]['function_id'] = -1
                            task_on_process_cloud[iot_index][cloud_index]['data_size'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['cpu_cycle'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['task_arrive_time'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['tran_finish_time'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['flag'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['remain'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['max_delay'] = 0
                            cloud_pool[cloud_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, cloud_pool[cloud_index][iot_index][id_index]])
                        else:
                            task_on_process_cloud[iot_index][cloud_index]['flag'] = 0
                            cloud_pool[cloud_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, cloud_pool[cloud_index][iot_index][id_index]])
                if slot_time_remain > 0 and task_on_process_cloud[iot_index][cloud_index]['remain'] > 0 and \
                        task_on_process_cloud[iot_index][cloud_index]['flag'] == 0:
                    id_index = task_on_process_cloud[iot_index][cloud_index]['function_id']
                    time_index = task_on_process_cloud[iot_index][cloud_index]['task_arrive_time']
                    if time_index == 0:
                        print("云取任务执行", slot_time_remain)
                    task_remain_time = task_on_process_cloud[iot_index][cloud_index]['max_delay'] - (
                                j + 1 - slot_time_remain - task_on_process_cloud[iot_index][cloud_index][
                            'task_arrive_time'])
                    if task_remain_time >= slot_time_remain:
                        task_on_process_cloud[iot_index][cloud_index]['remain'] = \
                        task_on_process_cloud[iot_index][cloud_index]['remain'] - comp_cap_cloud[
                            cloud_index] * slot_time_remain
                        if task_on_process_cloud[iot_index][cloud_index]['remain'] > 0:
                            slot_time_remain = 0
                            cloud_pool[cloud_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, cloud_pool[cloud_index][iot_index][id_index]])
                        if task_on_process_cloud[iot_index][cloud_index]['remain'] <= 0:
                            task_on_process_cloud[iot_index][cloud_index]['remain'] = \
                            task_on_process_cloud[iot_index][cloud_index]['remain'] + comp_cap_cloud[
                                cloud_index] * slot_time_remain
                            task_remain_need_time = task_on_process_cloud[iot_index][cloud_index]['remain'] / \
                                                    comp_cap_cloud[cloud_index]
                            task_on_process_cloud[iot_index][cloud_index]['remain'] = 0
                            slot_time_remain = slot_time_remain - task_remain_need_time
                            unfinish_indi[iot_index][time_index][0] = 0
                            computing_process_delay[iot_index][time_index][0] = \
                            task_on_process_cloud[iot_index][cloud_index]['cpu_cycle'] / comp_cap_cloud[cloud_index]
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                                   task_on_process_cloud[iot_index][cloud_index][
                                                                       'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * (comp_cap_cloud[cloud_index] ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                            task_on_process_cloud[iot_index][cloud_index]['function_id'] = -1
                            task_on_process_cloud[iot_index][cloud_index]['data_size'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['cpu_cycle'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['task_arrive_time'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['tran_finish_time'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['flag'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['remain'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['max_delay'] = 0
                            cloud_pool[cloud_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, cloud_pool[cloud_index][iot_index][id_index]])
                    if task_remain_time < slot_time_remain:
                        task_on_process_cloud[iot_index][cloud_index]['remain'] = \
                        task_on_process_cloud[iot_index][cloud_index]['remain'] - comp_cap_cloud[
                            cloud_index] * task_remain_time

                        if task_on_process_cloud[iot_index][cloud_index]['remain'] <= 0:
                            if task_on_process_cloud[iot_index][cloud_index]['remain'] == 0:
                                slot_time_remain = slot_time_remain - task_remain_time
                            if task_on_process_cloud[iot_index][cloud_index]['remain'] < 0:
                                task_on_process_cloud[iot_index][cloud_index]['remain'] = \
                                task_on_process_cloud[iot_index][cloud_index]['remain'] + comp_cap_cloud[
                                    cloud_index] * task_remain_time
                                task_remain_need_time = task_on_process_cloud[iot_index][cloud_index]['remain'] / \
                                                        comp_cap_cloud[cloud_index]
                                task_on_process_cloud[iot_index][cloud_index]['remain'] = 0
                                slot_time_remain = slot_time_remain - task_remain_need_time
                            unfinish_indi[iot_index][time_index][0] = 0
                            computing_process_delay[iot_index][time_index][0] = \
                            task_on_process_cloud[iot_index][cloud_index]['cpu_cycle'] / comp_cap_cloud[cloud_index]
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                                   task_on_process_cloud[iot_index][cloud_index][
                                                                       'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * (comp_cap_cloud[cloud_index] ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                            task_on_process_cloud[iot_index][cloud_index]['function_id'] = -1
                            task_on_process_cloud[iot_index][cloud_index]['data_size'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['cpu_cycle'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['task_arrive_time'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['tran_finish_time'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['flag'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['remain'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['max_delay'] = 0
                            cloud_pool[cloud_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, cloud_pool[cloud_index][iot_index][id_index]])
                        if task_on_process_cloud[iot_index][cloud_index]['remain'] > 0:
                            slot_time_remain = slot_time_remain - task_remain_time
                            unfinish_indi[iot_index][time_index][0] = 5
                            computing_process_delay[iot_index][time_index][0] = (task_on_process_cloud[iot_index][
                                                                                     cloud_index]['cpu_cycle'] -
                                                                                 task_on_process_cloud[iot_index][
                                                                                     cloud_index]['remain']) / \
                                                                                comp_cap_cloud[cloud_index]
                            delay_list[iot_index][time_index][0] = j + 1 - slot_time_remain - \
                                                                   task_on_process_cloud[iot_index][cloud_index][
                                                                       'task_arrive_time']
                            energy_list[iot_index][time_index][0] = (
                                        tr_power * transing_process_delay[iot_index][time_index][0]
                                        + esc * (comp_cap_cloud[cloud_index] ** 2) * ((
                                            computing_process_delay[iot_index][time_index][0] +
                                            cold_start_delay[iot_index][time_index][0])))
                            expense_list[iot_index][time_index][0] = 0.01 * 2 * \
                                                                     computing_process_delay[iot_index][time_index][
                                                                         0] + 6e-5
                            task_on_process_cloud[iot_index][cloud_index]['function_id'] = -1
                            task_on_process_cloud[iot_index][cloud_index]['data_size'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['cpu_cycle'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['task_arrive_time'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['tran_finish_time'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['flag'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['remain'] = 0
                            task_on_process_cloud[iot_index][cloud_index]['max_delay'] = 0
                            cloud_pool[cloud_index][iot_index][id_index] = np.max(
                                [j + 1 - slot_time_remain, cloud_pool[cloud_index][iot_index][id_index]])

    # 当前时隙结束：
    edge_count = np.zeros([1, n_edge])
    for edge_index in range(n_edge):
        for iot_index in range(n_iot):
            if (task_on_transmit_local[iot_index]['remain'] > 0 and task_on_transmit_local[iot_index][
                'iot_action'] - 1 == edge_index) or (all_task_wait_iot_tran_edge_count[iot_index][edge_index] != 0 and (
            not Queue_iot_tran[iot_index].empty())) or task_on_process_edge[iot_index][edge_index]['remain'] > 0 or (
            not Queue_edge_comp[iot_index][edge_index].empty()):
                edge_count[0][edge_index] += 1

    observation_all = np.zeros([n_iot, n_features - 5])
    for iot_index in range(n_iot):
        # n_features_system_iot
        for n_task_type_index in range(n_task_type):
            observation_all[iot_index, n_task_type_index] = all_task_wait_iot_comp_count[iot_index][n_task_type_index]
            observation_all[iot_index, n_task_type * 1 + n_task_type_index] = \
            per_task_wait_iot_comp_count_cpu_cycle[iot_index][n_task_type_index]
            observation_all[iot_index, n_task_type * 2 + n_task_type_index] = all_task_wait_iot_tran_count[iot_index][
                n_task_type_index]
            observation_all[iot_index, n_task_type * 3 + n_task_type_index] = \
            per_task_wait_iot_tran_count_size[iot_index][n_task_type_index]
            observation_all[iot_index, n_task_type * 4 + n_task_type_index] = np.min(
                [j + 1 - iot_pool[iot_index][n_task_type_index], keep_alive[n_task_type_index]])
        observation_all[iot_index, n_task_type * 5] = all_task_wait_iot_comp_cpu_cycle[iot_index]
        observation_all[iot_index, n_task_type * 5 + 1] = task_on_process_local[iot_index]['remain']
        observation_all[iot_index, n_task_type * 5 + 2] = all_task_wait_iot_tran_size[iot_index]
        observation_all[iot_index, n_task_type * 5 + 3] = task_on_transmit_local[iot_index]['remain']
        observation_all[iot_index, n_task_type * 5 + 4] = pull_uplink_rate_iot
        observation_all[iot_index, n_task_type * 5 + 5] = comp_cap_iot[iot_index]
        for i in range(n_edge):
            observation_all[iot_index, n_task_type * 5 + 6 + i] = tran_edge_cap_iot[iot_index][i]
        for i in range(n_cloud):
            observation_all[iot_index, n_task_type * 5 + 6 + n_edge + i] = tran_cloud_cap_iot[iot_index][i]

        observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud] = pull_uplink_rate_edge
        observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud + 1] = edge_count[0][0]
        observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud + 2] = \
        all_task_wait_edge_comp_size_cpu_cycle[iot_index][0]
        observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud + 3] = task_on_process_edge[iot_index][0][
            'remain']

        for n_task_type_index in range(n_task_type):
            observation_all[iot_index, n_task_type * 5 + 6 + n_edge + n_cloud + 4 + n_task_type_index] = \
            all_task_wait_edge_comp_count[iot_index][0][n_task_type_index]
            observation_all[iot_index, n_task_type * 6 + 6 + n_edge + n_cloud + 4 + n_task_type_index] = \
            per_task_wait_edge_comp_count_cpu_cycle[iot_index][0][n_task_type_index]
            temp_edge_pool = np.max(edge_pool[0][:, n_task_type_index])
            for iot_index in range(n_iot):
                edge_pool[0][iot_index][n_task_type_index] = temp_edge_pool
            observation_all[iot_index, n_task_type * 7 + 6 + n_edge + n_cloud + 4 + n_task_type_index] = np.min(
                [j + 1 - temp_edge_pool, keep_alive[n_task_type_index]])

        observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 4] = pull_uplink_rate_edge
        observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 5] = edge_count[0][1]
        observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 6] = \
        all_task_wait_edge_comp_size_cpu_cycle[iot_index][1]
        observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 7] = task_on_process_edge[iot_index][1][
            'remain']
        for n_task_type_index in range(n_task_type):
            observation_all[iot_index, n_task_type * 8 + 6 + n_edge + n_cloud + 8 + n_task_type_index] = \
            all_task_wait_edge_comp_count[iot_index][1][n_task_type_index]
            observation_all[iot_index, n_task_type * 9 + 6 + n_edge + n_cloud + 8 + n_task_type_index] = \
            per_task_wait_edge_comp_count_cpu_cycle[iot_index][1][n_task_type_index]
            temp_edge_pool = np.max(edge_pool[1][:, n_task_type_index])
            for iot_index in range(n_iot):
                edge_pool[1][iot_index][n_task_type_index] = temp_edge_pool
            observation_all[iot_index, n_task_type * 10 + 6 + n_edge + n_cloud + 8 + n_task_type_index] = np.min(
                [j + 1 - temp_edge_pool, keep_alive[n_task_type_index]])

        # n_features_system_cloud
        observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 8] = pull_uplink_rate_cloud
        observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 9] = comp_cap_cloud[0]
        observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 10] = \
        all_task_wait_cloud_comp_size_cpu_cycle[iot_index][0]
        observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 11] = task_on_process_cloud[iot_index][0][
            'remain']
        for n_task_type_index in range(n_task_type):
            observation_all[iot_index, n_task_type * 11 + 6 + n_edge + n_cloud + 12 + n_task_type_index] = \
            all_task_wait_cloud_comp_count[iot_index][0][n_task_type_index]
            observation_all[iot_index, n_task_type * 12 + 6 + n_edge + n_cloud + 12 + n_task_type_index] = \
            per_task_wait_cloud_comp_count_cpu_cycle[iot_index][0][n_task_type_index]
            temp_cloud_pool = np.max(cloud_pool[0][:, n_task_type_index])
            for iot_index in range(n_iot):
                cloud_pool[0][iot_index][n_task_type_index] = temp_cloud_pool

            observation_all[iot_index, n_task_type * 13 + 6 + n_edge + n_cloud + 12 + n_task_type_index] = np.min(
                [j + 1 - cloud_pool[0][iot_index][n_task_type_index], keep_alive[n_task_type_index]])
    return observation_all