import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from reset_step import reset, step


def reward(iot_id, j_time, type_id, time, energy, money, unfinish_indi, task, weight_1, weight_2, weight_3):
    max_energy = 0
    max_expense = 0
    t_id = int(type_id)
    max_delay_task = task[iot_id][j_time][0]['max_delay']
    if type_id != -1:
        if unfinish_indi == 0:
            reward1 = 20 * np.exp(-2 * time / max_delay_task) + 40 * t_id
        else:
            reward1 = -5 * np.exp(-2 * time / max_delay_task) - (
                    20 * np.exp(-2 * 0 / max_delay_task) * (t_id + 1) + 40 * t_id) - 10 * (t_id + unfinish_indi + 1)
    else:
        print(iot_id, j_time, type_id)
    reward2 = weight_1 * time + weight_2 * energy + money * weight_3  
    reward = reward1 - reward2  
    return reward

class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 n_time,
                 learning_rate=0.0001,
                 batch_size=64,
                 reward_decay=0.99,
                 alpha=0.1,
                 beta_increment=0.001,
                 e_greedy=0.99,
                 replace_target_iter=150,  
                 memory_size=3500 * 2,
                 e_greedy_increment=0.00005,
                 N_L1=64):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.alpha = alpha
        self.beta_increment = beta_increment
        self.beta = 0.004  
        self.epsilon = 0
        self.N_L1 = N_L1
        tf.reset_default_graph()
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.priority = np.zeros(self.memory_size)  
        self.memory_counter = 0
        self.target_params_replaced_counter = 0
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()
        self.store_q_value = list()
        self.loss_store = list()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        if (self.memory_counter > 200) and (self.memory_counter % 10 == 0):
            self.learn()
        
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.N_L1, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('lm1'):
                wm1 = tf.get_variable('wm1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bm1 = tf.get_variable('bm1', [1, n_l1], initializer=b_initializer, collections=c_names)
                lm1 = tf.nn.relu(tf.matmul(l1, wm1) + bm1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(lm1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('lm1'):
                wm1 = tf.get_variable('wm1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bm1 = tf.get_variable('bm1', [1, n_l1], initializer=b_initializer, collections=c_names)
                lm1 = tf.nn.relu(tf.matmul(l1, wm1) + bm1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(lm1, w2) + b2

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            self.target_params_replaced_counter = self.target_params_replaced_counter + 1
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  
                self.s: batch_memory[:, -self.n_features:],  
            })

        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        max_act4next = np.argmax(q_next, axis=1)  
        selected_q_next = q_next[batch_index, max_act4next]

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        
        choose_action_way_temp = -1
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})

            action_choose = np.argmax(actions_value, axis=1)
            
            choose_action_way_temp = 0
        else:
            
            action_choose = np.random.randint(0, self.n_actions)
            choose_action_way_temp = 1
       
        return action_choose, choose_action_way_temp


def train(
        iot_RL_list, NUM_EPISODE, n_iot, n_edge, n_cloud, n_task_type, n_time, n_features, keep_alive,
        pull_uplink_rate_iot, comp_cap_iot,
        tran_edge_cap_iot, tran_cloud_cap_iot,
        pull_uplink_rate_edge, pull_uplink_rate_cloud,
        comp_cap_cloud, comp_cap_edge,

        reward_indicator, task, weight_1, weight_2, weight_3,
        choose_action_way, n_actions, n_no_task,

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

        Queue_iot_comp, all_task_wait_iot_comp_count,
        per_task_wait_iot_comp_count_cpu_cycle, all_task_wait_iot_comp_cpu_cycle,
        Queue_iot_tran, all_task_wait_iot_tran_count, all_task_wait_iot_tran_edge_count,
        per_task_wait_iot_tran_count_size, all_task_wait_iot_tran_size,
        task_on_transmit_local, task_on_process_local,

        task_on_process_edge, wait_comp_process_delay, Queue_edge_comp, wait_trans_process_delay,
        esc, all_task_wait_edge_comp_count, per_task_wait_edge_comp_count_cpu_cycle,
        all_task_wait_edge_comp_size_cpu_cycle,

        Queue_cloud_comp, all_task_wait_cloud_comp_count,
        per_task_wait_cloud_comp_count_cpu_cycle, all_task_wait_cloud_comp_size_cpu_cycle,
        tr_power, task_on_process_cloud
):
    for episode in range(NUM_EPISODE):
        print("episode:", episode)
        observation_all_reset = reset(n_iot, n_edge, n_cloud, n_task_type, n_time, n_features, keep_alive,
                                      pull_uplink_rate_iot, comp_cap_iot,
                                      tran_edge_cap_iot, tran_cloud_cap_iot, pull_uplink_rate_edge,
                                      pull_uplink_rate_cloud, comp_cap_cloud, reward_indicator)
        observation_all = observation_all_reset
        for j in range(n_time):
            for iot_index in range(n_iot):
                observation_sys = np.squeeze(observation_all[iot_index, :])
                if task[iot_index][j][0]['function_id'] != -1:
                    observation_task = [task[iot_index][j][0]['function_id'], task[iot_index][j][0]['data_size'],
                                        task[iot_index][j][0]['cpu_cycle'],
                                        task[iot_index][j][0]['max_delay'],
                                        keep_alive[task[iot_index][j][0]['function_id']]
                                        ]
                    observation = np.hstack((observation_task, observation_sys))
                    history_observation[iot_index][j][0] = observation
                    update_index = []
                    for time_index in range(j):
                        if (1 - reward_indicator[iot_index][time_index][0]) * delay_list[iot_index][time_index][0] > 0:
                            update_index.append(time_index)
                    if len(update_index) != 0:
                        for update_ii in range(len(update_index)):
                            time_index = update_index[update_ii]
                            id_index = int(task[iot_index][time_index][0]['function_id'])
                            if id_index == -1:
                                print("update_index", update_index, j, time_index)
                                print("action[iot_index][time_index][0]", action[iot_index][time_index][0])
                                print("reward_indicator[iot_index][time_index][0]",
                                      reward_indicator[iot_index][time_index][0])
                                print("delay_list[iot_index][time_index][0]", delay_list[iot_index][time_index][0])
                            reward_list[iot_index][time_index][0] = reward(iot_index, time_index, id_index,
                                                                           delay_list[iot_index][time_index][0],
                                                                           energy_list[iot_index][time_index][0],
                                                                           expense_list[iot_index][time_index][0],
                                                                           unfinish_indi[iot_index][time_index][0],
                                                                           task, weight_1, weight_2, weight_3)
                            iot_RL_list[iot_index].store_transition(history_observation[iot_index][time_index][0],
                                                                    action[iot_index][time_index][0],
                                                                    reward_list[iot_index][time_index][0],
                                                                    observation)
                            reward_indicator[iot_index][time_index][0] = 1
                    action[iot_index][j][0], choose_action_way[iot_index][j][0] = iot_RL_list[iot_index].choose_action(
                        observation)
                    if action[iot_index][j][0] < 0 or action[iot_index][j][0] > n_actions - 1:
                        print("look_task_time_index")
                if task[iot_index][j][0]['function_id'] == -1 and j > n_time - n_no_task:
                    look_task_time_index = 0
                    while look_task_time_index < n_time:
                        if task[iot_index][look_task_time_index][0]['function_id'] != -1:
                            observation_task = [task[iot_index][look_task_time_index][0]['function_id'],
                                                task[iot_index][look_task_time_index][0]['data_size'],
                                                task[iot_index][look_task_time_index][0]['cpu_cycle'],
                                                task[iot_index][look_task_time_index][0]['max_delay'],
                                                keep_alive[task[iot_index][look_task_time_index][0]['function_id']]
                                                ]
                            break
                        else:
                            look_task_time_index += 1
                    observation = history_observation[iot_index][look_task_time_index][0]
                    
                    update_index = []
                    for time_index in range(j):
                        if (1 - reward_indicator[iot_index][time_index][0]) * delay_list[iot_index][time_index][0] > 0:
                            update_index.append(time_index)
                    if len(update_index) != 0:
                        for update_ii in range(len(update_index)):
                            time_index = update_index[update_ii]
                            id_index = int(task[iot_index][time_index][0]['function_id'])
                            reward_list[iot_index][time_index][0] = reward(iot_index, time_index, id_index,
                                                                           delay_list[iot_index][time_index][0],
                                                                           energy_list[iot_index][time_index][0],
                                                                           expense_list[iot_index][time_index][0],
                                                                           unfinish_indi[iot_index][time_index][0],
                                                                           task, weight_1, weight_2, weight_3)
                            iot_RL_list[iot_index].store_transition(history_observation[iot_index][time_index][0],
                                                                    action[iot_index][time_index][0],
                                                                    reward_list[iot_index][time_index][0],
                                                                    observation)
                            reward_indicator[iot_index][time_index][0] = 1

                if task[iot_index][j][0]['function_id'] == -1:
                    action[iot_index][j][0] = -2
                if action[iot_index][j][0] == -1:
                    print(iot_index, j, observation[0])

            observation_all_ = step(
                j, action,
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

                tr_power, task_on_process_cloud
            )

            observation_all = observation_all_
        unfinish_indi_count = 0
        unfinish_indi0_count = 0
        unfinish_indi1_count = 0
        unfinish_indi2_count = 0
        unfinish_indi3_count = 0
        unfinish_indi4_count = 0
        unfinish_indi5_count = 0
        reward_list_count = 0
        finish_indi_count = 0
        non_indi_count = 0
        iot_a_result = 0
        edge_a_result = 0
        cloud_a_result = 0
        total_delay = 0
        total_energy = 0
        total_expense = 0
        cold_start_result = 0
        warm_start_result = 0
        cold_start_non = 0
        total_cold_start = 0
        total_transing_process = 0
        total_computing_process = 0
        action_non = 0
        choose_action_by_random_count = 0
        choose_action_by_net_count = 0
        for i in range(n_iot):
            unfinish_indi_count = unfinish_indi_count + (np.sum(unfinish_indi[i] != 0) - np.sum(unfinish_indi[i] == -1))
            finish_indi_count = finish_indi_count + np.sum(unfinish_indi[i] == 0)
            non_indi_count = non_indi_count + np.sum(unfinish_indi[i] == -1)
            unfinish_indi0_count = unfinish_indi0_count + np.sum(unfinish_indi[i] == 0)
            unfinish_indi1_count = unfinish_indi1_count + np.sum(unfinish_indi[i] == 1)
            unfinish_indi2_count = unfinish_indi2_count + np.sum(unfinish_indi[i] == 2)
            unfinish_indi3_count = unfinish_indi3_count + np.sum(unfinish_indi[i] == 3)
            unfinish_indi4_count = unfinish_indi4_count + np.sum(unfinish_indi[i] == 4)
            unfinish_indi5_count = unfinish_indi5_count + np.sum(unfinish_indi[i] == 5)

            reward_list_count = reward_list_count + np.sum(reward_list[i])

            iot_a_result = iot_a_result + np.sum(action[i] == 0)
            edge_a_result = edge_a_result + np.sum(action[i] == 1) + np.sum(action[i] == 2)
            cloud_a_result = cloud_a_result + np.sum(action[i] == 3)
            action_non = action_non + np.sum(action[i] == -1)

            total_delay = total_delay + np.sum(delay_list[i])
            total_energy = total_energy + np.sum(energy_list[i])
            total_expense = total_expense + np.sum(expense_list[i])

            cold_start_result = cold_start_result + np.sum(cold_count_list[i] == 1)
            warm_start_result = warm_start_result + np.sum(cold_count_list[i] == 0)

            cold_start_non = cold_start_non + np.sum(cold_count_list[i] == -1)
            total_cold_start = total_cold_start + np.sum(cold_start_delay[i])

            total_transing_process = total_transing_process + np.sum(transing_process_delay[i])
            total_computing_process = total_computing_process + np.sum(computing_process_delay[i])
            choose_action_by_random_count = choose_action_by_random_count + np.sum(choose_action_way[i] == 1)
            choose_action_by_net_count = np.sum(choose_action_way[i] == 0)
        reward_list_record.append(reward_list_count)
        delay_list_record.append(total_delay)
        energy_list_record.append(total_energy)
        expense_list_record.append(total_expense)
        energy_avg_list_record = []
        expense_avg_list_record = []
        unfinish_total_list_record.append(unfinish_indi_count)
        unfinish_total_list_record0.append(unfinish_indi0_count)
        unfinish_total_list_record1.append(unfinish_indi1_count)
        unfinish_total_list_record2.append(unfinish_indi2_count)
        unfinish_total_list_record3.append(unfinish_indi3_count)
        unfinish_total_list_record4.append(unfinish_indi4_count)
        unfinish_total_list_record5.append(unfinish_indi5_count)

        utilization_ratio_record = []  
        loss_list_record = []  
        action_record_iot.append(iot_a_result)
        action_record_edge.append(edge_a_result)
        action_record_cloud.append(cloud_a_result)
        choose_action_by_random.append(choose_action_by_random_count)
        choose_action_by_net.append(choose_action_by_net_count)
        slot_bandwidth_record = []
        cold_count_list_record.append(cold_start_result)
        warm_count_list_record.append(warm_start_result)
        unfinish_indi_record = []
        sucess_count_list_record.append(finish_indi_count)

        np.savetxt(f'../结果分析/奖励-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
                   reward_list_record, delimiter=',')
        np.savetxt(
            f'../结果分析/丢失任务总数-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            unfinish_total_list_record, delimiter=',')
        np.savetxt(
            f'../结果分析/任务总时延-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            delay_list_record, delimiter=',')
        np.savetxt(
            f'../结果分析/任务总能耗-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            energy_list_record, delimiter=',')
        np.savetxt(
            f'../结果分析/任务总花费-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            expense_list_record, delimiter=',')
        np.savetxt(
            f'../结果分析/任务是冷启动-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            cold_count_list_record, delimiter=',')
        np.savetxt(
            f'../结果分析/任务是热启动-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            warm_count_list_record, delimiter=',')
        np.savetxt(
            f'../结果分析/成功任务总数-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            sucess_count_list_record, delimiter=',')
        np.savetxt(
            f'../结果分析/卸载结果iot_-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            action_record_iot, delimiter=',')
        np.savetxt(
            f'../结果分析/卸载结果edge_-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            action_record_edge, delimiter=',')
        np.savetxt(
            f'../结果分析/卸载结果cloud_-DQN_4_arrive_0001_learning_99_discount_64_batchsize_x1_0_cold_delay_4_iot.csv',
            action_record_cloud, delimiter=',')
