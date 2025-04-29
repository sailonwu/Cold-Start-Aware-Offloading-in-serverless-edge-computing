
class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 n_time,
                 learning_rate=0.001,
                 batch_size=64,
                 reward_decay=0.99,
                 alpha = 0.1,
                 beta_increment = 0.001, 
                 e_greedy=0.99,
                 replace_target_iter=150,#400-->200-->100
                 memory_size=3500*2,
                 e_greedy_increment=0.00005,
                 N_L1 = 64):
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
        self.beta = 0.004  # initial value of beta
        self.epsilon = 0
        self.N_L1 = N_L1
        
        
        tf.reset_default_graph()

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.priority = np.zeros(self.memory_size)  # Initialize priorities
        
        
        self.memory_counter = 0
        self.target_params_replaced_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.cost_his = []
        
        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()
        self.store_q_value = list()
        self.loss_store = list()
        
    
    def save_model(self, path):
        self.saver.save(self.sess, path)
        #print(f"Model saved to: {path}")

    def load_model(self, path):
        self.saver.restore(self.sess, path)
        print(f"Model restored from: {path}")
    def store_transition(self, s, a, r, s_):
        ##print(np.shape(s),np.shape(s_))
        #print(np.shape(s),np.shape(s_))
        #print(s)
        #print(a)
        #print(r)
        #print(s_)
        #print()
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        #print("向回放区存入经验的次数：self.memory_counter：",self.memory_counter)
        if (self.memory_counter > 200) and (self.memory_counter % 10 == 0):
            self.learn()
            #print("self.learn_step_counter",self.learn_step_counter)
        #print(transition[:self.n_features])
        #print(transition[self.n_features],transition[self.n_features+1])
        #print(transition[-self.n_features:])

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.N_L1,  \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # hidden layer. collections is used later when assign to target net
            with tf.variable_scope('lm1'):
                wm1 = tf.get_variable('wm1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bm1 = tf.get_variable('bm1', [1, n_l1], initializer=b_initializer, collections=c_names)
                lm1 = tf.nn.relu(tf.matmul(l1, wm1) + bm1)

            # Value stream
            with tf.variable_scope('Value'):
                w2_V  = tf.get_variable('w2_V', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2_V  = tf.get_variable('b2_V', [1, 1], initializer=b_initializer, collections=c_names)
                self.V = tf.matmul(lm1, w2_V ) + b2_V

            # Advantage stream
            with tf.variable_scope('Advantage'):
                w2_A  = tf.get_variable('w2_A', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2_A  = tf.get_variable('b2_A', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(lm1, w2_A ) + b2_A 

            with tf.variable_scope('Q'):
                self.q_eval = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # hidden layer. collections is used later when assign to target net
            with tf.variable_scope('lm1'):
                wm1 = tf.get_variable('wm1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bm1 = tf.get_variable('bm1', [1, n_l1], initializer=b_initializer, collections=c_names)
                lm1 = tf.nn.relu(tf.matmul(l1, wm1) + bm1)

            # Value stream
            with tf.variable_scope('Value'):
                w2_V  = tf.get_variable('w2_V', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2_V  = tf.get_variable('b2_V', [1, 1], initializer=b_initializer, collections=c_names)
                self.V_ = tf.matmul(lm1, w2_V ) + b2_V 

            # Advantage stream
            with tf.variable_scope('Advantage'):
                w2_A = tf.get_variable('w2_A', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2_A = tf.get_variable('b2_A', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.A_ = tf.matmul(lm1, w2_A) + b2_A 

            with tf.variable_scope('Q'):
                self.q_next = self.V_ + (self.A_ - tf.reduce_mean(self.A_, axis=1, keepdims=True))
    
    def learn(self):
        # check to replace target parameters超过100步，每十步学一次，每200步更新目标网络参数，更新的步长得是学习步长得整数倍，不然就错过网络更新环节
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            self.target_params_replaced_counter = self.target_params_replaced_counter + 1
            #print('\ntarget_params_replaced','学习总步长', self.learn_step_counter,'\n')
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # Evaluate q_next and q_eval using target_net and eval_net
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # next state
                self.s: batch_memory[:, -self.n_features:],  # current state
            })

        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        # Change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        #DDQN算法在估计回报时只需要用 评估网络确定动作，用 目标网络确定动作价值 即可
        max_act4next = np.argmax(q_eval4next, axis=1)  # the action that maximizes the next Q value
        #selected_q_next = q_eval4next[batch_index, max_act4next]    #DQN
        selected_q_next = q_next[batch_index, max_act4next]        # Double DQN
        

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # Train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                    feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # Increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1 

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        #action_choose = np.random.randint(0, self.n_actions)
        #print()
        choose_action_way_temp = -1
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            
            action_choose = np.argmax(actions_value, axis=1)
            #print("",action_choose)
            #print("网络选择动作,actions_value",actions_value,action_choose)
            choose_action_way_temp = 0
        else:
            #print("随机选择动作")
            action_choose = np.random.randint(0, self.n_actions)
            choose_action_way_temp = 1
        #action_choose = np.random.randint(0, 4)
        #action_choose = 3
        #print("        选择动作",action_choose)
        return action_choose, choose_action_way_temp

if __name__ == "__main__":
    task = []
    for i in range(n_iot):
        with open(f'任务列表\iot_{i}_trace_6_new.json', 'r') as f:
            task.append(json.load(f))

    total_task = 0  
    total_task_non = 0                                              

    for i in range(n_iot):
        for j in range(n_time):
            if j <= (n_time - n_no_task) :
                
                if task[i][j][0]['function_id'] != -1:
                        total_task = total_task +1
                else:
                        
                        total_task_non = total_task_non + 1
            else:
                #for t in range(n_task_type):
                    task[i][j][0]['function_id'] = -1
                    task[i][j][0]['data_size'] = 0
                    task[i][j][0]['cpu_cycle'] = 0
                    total_task_non = total_task_non + 1

    print("任务请求总数量、non：7002 2998",total_task,total_task_non)
    with open('任务数txt\\record-load_load_0_Dou_Due_6_arrive_001_learning_99_discount_64_batchsize_max_delay_4_iot.txt', 'w') as f:
        f.write(f"[{total_task}, {total_task_non}]\n")
    
    
    iot_RL_list1 = list()
    for i in range(n_iot):
        net = DeepQNetwork(n_actions, n_features, n_time)
        net.load_model(f"Dou_Due_6_arrive_001_learning_99_discount_64_batchsize_max_delay_4_iot_iot_agent_0_9800.ckpt")
        iot_RL_list1.append(net)   
    train(iot_RL_list1, NUM_EPISODE)
    print('1_Training Finished') 
     
    
